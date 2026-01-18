from .basestemmer import BaseStemmer
from .among import Among
def __r_residual_form(self):
    self.ket = self.cursor
    among_var = self.find_among_b(PortugueseStemmer.a_8)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.__r_RV():
            return False
        if not self.slice_del():
            return False
        self.ket = self.cursor
        try:
            v_1 = self.limit - self.cursor
            try:
                if not self.eq_s_b(u'u'):
                    raise lab1()
                self.bra = self.cursor
                v_2 = self.limit - self.cursor
                if not self.eq_s_b(u'g'):
                    raise lab1()
                self.cursor = self.limit - v_2
                raise lab0()
            except lab1:
                pass
            self.cursor = self.limit - v_1
            if not self.eq_s_b(u'i'):
                return False
            self.bra = self.cursor
            v_3 = self.limit - self.cursor
            if not self.eq_s_b(u'c'):
                return False
            self.cursor = self.limit - v_3
        except lab0:
            pass
        if not self.__r_RV():
            return False
        if not self.slice_del():
            return False
    elif not self.slice_from(u'c'):
        return False
    return True