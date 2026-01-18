from .basestemmer import BaseStemmer
from .among import Among
def __r_KER(self):
    if not self.out_grouping(IndonesianStemmer.g_vowel, 97, 117):
        return False
    if not self.eq_s(u'er'):
        return False
    return True