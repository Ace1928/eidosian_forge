import re
from nltk.stem.api import StemmerI
def __isAcceptable(self, word, remove_total):
    """Determine if the word is acceptable for stemming."""
    word_is_acceptable = False
    if word[0] in 'aeiouy':
        if len(word) - remove_total >= 2:
            word_is_acceptable = True
    elif len(word) - remove_total >= 3:
        if word[1] in 'aeiouy':
            word_is_acceptable = True
        elif word[2] in 'aeiouy':
            word_is_acceptable = True
    return word_is_acceptable