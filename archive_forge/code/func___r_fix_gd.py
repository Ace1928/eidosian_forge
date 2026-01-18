from .basestemmer import BaseStemmer
from .among import Among
def __r_fix_gd(self):
    self.ket = self.cursor
    if self.find_among_b(LithuanianStemmer.a_4) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_from(u'g'):
        return False
    return True