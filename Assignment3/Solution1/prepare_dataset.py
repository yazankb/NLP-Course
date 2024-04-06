import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

entity_set = set()

def transform_data(data):
    text = data['sentences']
    sentences = sent_tokenize(text)  # Tokenize the text into sentences
    transformed_data = []
    offset = 0  # Character offset to adjust NER indices
    
    for i, sentence in enumerate(sentences):
        tokenized_text = word_tokenize(sentence, language= 'russian')
        if not tokenized_text:
            print(f"No tokens found for sentence: '{sentence}' at index {i}")
            continue
        
        sentence_start = text.find(sentence, offset)
        if sentence_start == -1:
            print(f"Sentence not found in text: '{sentence}'")
            continue

        # Mapping from character indices to token indices within this sentence
        char_to_token_index = {}
        pos = sentence_start
        for index, token in enumerate(tokenized_text):
            token_pos = text.find(token, pos)
            if token_pos == -1:
                while (text[pos] == ' '):
                    pos +=1
                if text[pos] == '"':
                    token = '"'
                else:
                    token = "''"
                token_pos = pos
            for j in range(token_pos, token_pos + len(token)):
                char_to_token_index[j] = index
            pos = token_pos + len(token)

        # Transforming entities to the new token indices for this sentence
        entities = []
        for start_char, end_char, ent_type in data['ners']:
            if start_char >= sentence_start and end_char < sentence_start + len(sentence):
                start_token_index = char_to_token_index.get(start_char, None)
                end_token_index = char_to_token_index.get(end_char, None)
                if start_token_index is not None and end_token_index is not None:
                    entity_set.add(ent_type)
                    entities.append({
                        'type': ent_type,
                        'start': start_token_index,
                        'end': end_token_index + 1  # Ensure inclusivity
                    })

        # Setting ltokens and rtokens for contextual tokens
        ltokens = word_tokenize(sentences[i-1]) if i > 0 else []
        rtokens = word_tokenize(sentences[i+1]) if i < len(sentences) - 1 else []

        if entities:
            transformed_data.append({
                'tokens': tokenized_text,
                'entities': entities,
                'relations': [],
                'orig_id': str(data['id']),
                'ltokens': ltokens,
                'rtokens': rtokens
            })

        offset = sentence_start + len(sentence)  # Update offset to after this sentence

    return transformed_data

# Loading input data
input_data = []
with open("train.jsonl", 'r', encoding='utf-8') as file:  # Specify encoding here
    for line in file:
        input_data.append(json.loads(line))

frac = 0.1  # fraction of data to use for validation
validation_data = input_data[:int(frac * len(input_data))]
training_data = input_data[int(frac * len(input_data)):]

# Transform data to correct format
validation_data = [item for sublist in validation_data for item in transform_data(sublist) if item and item['tokens']]
training_data = [item for sublist in training_data for item in transform_data(sublist) if item and item['tokens']]

# Saving the transformed data to JSON files
with open('runne_val.json', 'w', encoding='utf-8') as json_file:  # Specify encoding here
    json.dump(validation_data, json_file)

with open('runne_train.json', 'w', encoding='utf-8') as json_file:  # Specify encoding here
    json.dump(training_data, json_file)

entities_dict = {}
for entity in entity_set:
    entities_dict[entity] = {"short": entity, "verbose": entity}

data_structure = {
    "entities": entities_dict,
    "relations": {}
}

with open('runne_types.json', 'w', encoding='utf-8') as json_file:  # Specify encoding here
    json.dump(data_structure, json_file, indent=4)

print("JSON file 'entities.json' has been created with the following content:")
print(json.dumps(data_structure, indent=4))

with open('runne_val.json', 'w') as json_file:
    json.dump(validation_data, json_file)

# Save training_data to a JSON file
with open('runne_train.json', 'w') as json_file:
    json.dump(training_data, json_file)

