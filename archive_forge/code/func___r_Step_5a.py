from .basestemmer import BaseStemmer
from .among import Among
def __r_Step_5a(self):
    self.ket = self.cursor
    if not self.eq_s_b(u'e'):
        return False
    self.bra = self.cursor
    try:
        v_1 = self.limit - self.cursor
        try:
            if not self.__r_R2():
                raise lab1()
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        if not self.__r_R1():
            return False
        v_2 = self.limit - self.cursor
        try:
            if not self.__r_shortv():
                raise lab2()
            return False
        except lab2:
            pass
        self.cursor = self.limit - v_2
    except lab0:
        pass
    if not self.slice_del():
        return False
    return True