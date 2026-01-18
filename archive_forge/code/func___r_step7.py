from .basestemmer import BaseStemmer
from .among import Among
def __r_step7(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_67) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    return True