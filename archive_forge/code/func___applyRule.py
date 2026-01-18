import re
from nltk.stem.api import StemmerI
def __applyRule(self, word, remove_total, append_string):
    """Apply the stemming rule to the word"""
    new_word_length = len(word) - remove_total
    word = word[0:new_word_length]
    if append_string:
        word += append_string
    return word