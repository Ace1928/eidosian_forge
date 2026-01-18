from .basestemmer import BaseStemmer
from .among import Among
def __r_step_0(self):
    self.ket = self.cursor
    among_var = self.find_among_b(RomanianStemmer.a_1)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if not self.__r_R1():
        return False
    if among_var == 1:
        if not self.slice_del():
            return False
    elif among_var == 2:
        if not self.slice_from(u'a'):
            return False
    elif among_var == 3:
        if not self.slice_from(u'e'):
            return False
    elif among_var == 4:
        if not self.slice_from(u'i'):
            return False
    elif among_var == 5:
        v_1 = self.limit - self.cursor
        try:
            if not self.eq_s_b(u'ab'):
                raise lab0()
            return False
        except lab0:
            pass
        self.cursor = self.limit - v_1
        if not self.slice_from(u'i'):
            return False
    elif among_var == 6:
        if not self.slice_from(u'at'):
            return False
    elif not self.slice_from(u'aţi'):
        return False
    return True