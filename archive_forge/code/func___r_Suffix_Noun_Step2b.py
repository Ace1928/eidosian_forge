from .basestemmer import BaseStemmer
from .among import Among
def __r_Suffix_Noun_Step2b(self):
    self.ket = self.cursor
    if self.find_among_b(ArabicStemmer.a_13) == 0:
        return False
    self.bra = self.cursor
    if not len(self.current) >= 5:
        return False
    if not self.slice_del():
        return False
    return True