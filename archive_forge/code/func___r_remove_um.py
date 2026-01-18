from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_um(self):
    self.B_found_a_match = False
    if not self.__r_has_min_length():
        return False
    self.limit_backward = self.cursor
    self.cursor = self.limit
    self.ket = self.cursor
    if not self.eq_s_b(u'ும்'):
        return False
    self.bra = self.cursor
    if not self.slice_from(u'்'):
        return False
    self.B_found_a_match = True
    self.cursor = self.limit_backward
    v_1 = self.cursor
    self.__r_fix_ending()
    self.cursor = v_1
    return True