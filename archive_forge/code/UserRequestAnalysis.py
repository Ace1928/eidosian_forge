import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk, tree2conlltags
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class Topic:
    def __init__(self, name: str, category: str = None, significance: float = 0.0, position: int = 0, associated_phrases: List[str] = [], relationships: List[Tuple[str, str]] = []):
        self.name = name
        self.category = category
        self.significance = significance
        self.position = position
        self.associated_phrases = associated_phrases
        self.relationships = relationships

def preprocess(text: str) -> List[tuple]:
    """
    Preprocess the text by tokenizing, lemmatizing, and part-of-speech tagging.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return pos_tag(lemmatized_tokens)

def semantic_similarity(word1: str, word2: str) -> float:
    """
    Calculate semantic similarity between two words using WordNet.
    """
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if synsets1 and synsets2:
        return max((s1.path_similarity(s2) or 0) for s1 in synsets1 for s2 in synsets2)
    return 0.0

def infer_relationships(tagged_sentence: List[tuple], topics: List[Topic], full_text: str) -> List[Topic]:
    """
    Infer relationships between topics based on their context and position in the full text.
    """
    for topic in topics:
        for other_topic in topics:
            if topic != other_topic:
                # Advanced inference considering thematic connections
                if any(word in full_text for word in topic.associated_phrases) and any(word in full_text for word in other_topic.associated_phrases):
                    topic.relationships.append((other_topic.name, "thematic connection"))
    return topics

def extract_named_entities(tagged_sentence: List[tuple]) -> List[Topic]:
    """
    Extract named entities from the tagged sentence, categorize, score, and contextualize them.
    """
    chunked = ne_chunk(tagged_sentence)
    iob_tagged = tree2conlltags(chunked)
    topics = defaultdict(list)
    word_freq = Counter([word for word, tag in tagged_sentence])

    for index, (word, tag, iob) in enumerate(iob_tagged):
        if iob != 'O':
            entity_type = iob.split('-')[1]
            topics[entity_type].append((word, index))

    processed_topics = []
    for category, entities in topics.items():
        for entity, position in entities:
            significance = word_freq[entity] * semantic_similarity(entity, category)
            context_window = tagged_sentence[max(0, position-2):min(len(tagged_sentence), position+3)]
            associated_phrases = [word for word, _ in context_window]
            processed_topics.append(Topic(entity, category, significance, position, associated_phrases))

    return processed_topics

def DissectUserRequest(request: str) -> List[Topic]:
    """
    Dissect user request into topics, considering context, relevance, significance, position, and inter-topic thematic relationships.
    """
    preprocessed_text = preprocess(request)
    topics = extract_named_entities(preprocessed_text)
    topics = infer_relationships(preprocessed_text, topics, request)
    return sorted(topics, key=lambda t: t.significance, reverse=True)

# Example usage
user_request = "Discuss the impact of AI on healthcare and future space missions."
topics = DissectUserRequest(user_request)
for topic in topics:
    print(f"Topic: {topic.name}, Category: {topic.category}, Significance: {topic.significance}, Position: {topic.position}, Associated Phrases: {topic.associated_phrases}, Relationships: {topic.relationships}")
