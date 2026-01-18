from .basestemmer import BaseStemmer
from .among import Among
def __r_step6(self):
    v_1 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if self.find_among_b(GreekStemmer.a_65) == 0:
            raise lab0()
        self.bra = self.cursor
        if not self.slice_from(u'μα'):
            return False
    except lab0:
        pass
    self.cursor = self.limit - v_1
    if not self.B_test1:
        return False
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_66) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    return True