import re
from nltk.stem.api import StemmerI
def _contains_vowel(self, stem):
    """Returns True if stem contains a vowel, else False"""
    for i in range(len(stem)):
        if not self._is_consonant(stem, i):
            return True
    return False