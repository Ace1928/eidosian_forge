from .basestemmer import BaseStemmer
from .among import Among
def __r_e_ending(self):
    self.B_e_found = False
    self.ket = self.cursor
    if not self.eq_s_b(u'e'):
        return False
    self.bra = self.cursor
    if not self.__r_R1():
        return False
    v_1 = self.limit - self.cursor
    if not self.out_grouping_b(DutchStemmer.g_v, 97, 232):
        return False
    self.cursor = self.limit - v_1
    if not self.slice_del():
        return False
    self.B_e_found = True
    if not self.__r_undouble():
        return False
    return True