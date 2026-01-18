import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk, tree2conlltags
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
def DissectUserRequest(request: str) -> List[Topic]:
    """
    Dissect user request into topics, considering context, relevance, significance, position, and inter-topic thematic relationships.
    """
    preprocessed_text = preprocess(request)
    topics = extract_named_entities(preprocessed_text)
    topics = infer_relationships(preprocessed_text, topics, request)
    return sorted(topics, key=lambda t: t.significance, reverse=True)