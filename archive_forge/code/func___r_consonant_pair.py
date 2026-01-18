from .basestemmer import BaseStemmer
from .among import Among
def __r_consonant_pair(self):
    v_1 = self.limit - self.cursor
    if self.cursor < self.I_p1:
        return False
    v_3 = self.limit_backward
    self.limit_backward = self.I_p1
    self.ket = self.cursor
    if self.find_among_b(NorwegianStemmer.a_1) == 0:
        self.limit_backward = v_3
        return False
    self.bra = self.cursor
    self.limit_backward = v_3
    self.cursor = self.limit - v_1
    if self.cursor <= self.limit_backward:
        return False
    self.cursor -= 1
    self.bra = self.cursor
    if not self.slice_del():
        return False
    return True