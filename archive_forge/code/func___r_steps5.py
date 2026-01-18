from .basestemmer import BaseStemmer
from .among import Among
def __r_steps5(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_11) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.B_test1 = False
    self.ket = self.cursor
    self.bra = self.cursor
    among_var = self.find_among_b(GreekStemmer.a_10)
    if among_var == 0:
        return False
    if self.cursor > self.limit_backward:
        return False
    if among_var == 1:
        if not self.slice_from(u'ι'):
            return False
    elif not self.slice_from(u'ιστ'):
        return False
    return True