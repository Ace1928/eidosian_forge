import spacy

# Load the pre-trained English model
nlp = spacy.load("en_core_web_sm")

# Documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy cat.",
    "The quick brown fox is jumping over the lazy dog.",
]

# Calculate similarity between each pair of documents
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        doc1 = nlp(documents[i])
        doc2 = nlp(documents[j])
        similarity = doc1.similarity(doc2)
        print(f"Similarity between Document {i+1} and Document {j+1}: {similarity}")
