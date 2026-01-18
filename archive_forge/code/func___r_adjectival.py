from .basestemmer import BaseStemmer
from .among import Among
def __r_adjectival(self):
    if not self.__r_adjective():
        return False
    v_1 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        among_var = self.find_among_b(RussianStemmer.a_2)
        if among_var == 0:
            self.cursor = self.limit - v_1
            raise lab0()
        self.bra = self.cursor
        if among_var == 1:
            try:
                v_2 = self.limit - self.cursor
                try:
                    if not self.eq_s_b(u'а'):
                        raise lab2()
                    raise lab1()
                except lab2:
                    pass
                self.cursor = self.limit - v_2
                if not self.eq_s_b(u'я'):
                    self.cursor = self.limit - v_1
                    raise lab0()
            except lab1:
                pass
            if not self.slice_del():
                return False
        elif not self.slice_del():
            return False
    except lab0:
        pass
    return True