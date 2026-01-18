from .basestemmer import BaseStemmer
from .among import Among
def __r_Step_4(self):
    self.ket = self.cursor
    among_var = self.find_among_b(EnglishStemmer.a_7)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if not self.__r_R2():
        return False
    if among_var == 1:
        if not self.slice_del():
            return False
    else:
        try:
            v_1 = self.limit - self.cursor
            try:
                if not self.eq_s_b(u's'):
                    raise lab1()
                raise lab0()
            except lab1:
                pass
            self.cursor = self.limit - v_1
            if not self.eq_s_b(u't'):
                return False
        except lab0:
            pass
        if not self.slice_del():
            return False
    return True