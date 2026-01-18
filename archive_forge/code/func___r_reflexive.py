from .basestemmer import BaseStemmer
from .among import Among
def __r_reflexive(self):
    self.ket = self.cursor
    if self.find_among_b(RussianStemmer.a_3) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    return True