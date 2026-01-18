from .basestemmer import BaseStemmer
from .among import Among
def __r_steps4(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_9) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.B_test1 = False
    self.ket = self.cursor
    self.bra = self.cursor
    if self.find_among_b(GreekStemmer.a_8) == 0:
        return False
    if self.cursor > self.limit_backward:
        return False
    if not self.slice_from(u'ι'):
        return False
    return True