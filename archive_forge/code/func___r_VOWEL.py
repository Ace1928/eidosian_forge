from .basestemmer import BaseStemmer
from .among import Among
def __r_VOWEL(self):
    if not self.in_grouping(IndonesianStemmer.g_vowel, 97, 117):
        return False
    return True