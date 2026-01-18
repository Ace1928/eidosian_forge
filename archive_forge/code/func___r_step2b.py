from .basestemmer import BaseStemmer
from .among import Among
def __r_step2b(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_26) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.ket = self.cursor
    self.bra = self.cursor
    if self.find_among_b(GreekStemmer.a_27) == 0:
        return False
    if not self.slice_from(u'εδ'):
        return False
    return True