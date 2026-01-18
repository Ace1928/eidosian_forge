import spacy

# Load the pre-trained English model
nlp = spacy.load("en_core_web_sm")

# News articles
articles = [
    "Apple Inc. is planning to launch a new iPhone model next month. The company's CEO, Tim Cook, made the announcement during a press conference in Cupertino, California.",
    "The United Nations held a summit on climate change in New York City. Leaders from various countries, including the United States, China, and India, attended the event.",
]

# Perform named entity recognition on each article
for article in articles:
    doc = nlp(article)

    print("Article:", article)
    print("Named Entities:")
    for entity in doc.ents:
        print(f"- {entity.text} ({entity.label_})")
    print("---")
