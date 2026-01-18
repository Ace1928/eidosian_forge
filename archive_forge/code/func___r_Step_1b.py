from .basestemmer import BaseStemmer
from .among import Among
def __r_Step_1b(self):
    self.ket = self.cursor
    among_var = self.find_among_b(EnglishStemmer.a_4)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.__r_R1():
            return False
        if not self.slice_from(u'ee'):
            return False
    else:
        v_1 = self.limit - self.cursor
        if not self.go_out_grouping_b(EnglishStemmer.g_v, 97, 121):
            return False
        self.cursor -= 1
        self.cursor = self.limit - v_1
        if not self.slice_del():
            return False
        v_2 = self.limit - self.cursor
        among_var = self.find_among_b(EnglishStemmer.a_3)
        if among_var == 0:
            return False
        self.cursor = self.limit - v_2
        if among_var == 1:
            c = self.cursor
            self.insert(self.cursor, self.cursor, u'e')
            self.cursor = c
        elif among_var == 2:
            self.ket = self.cursor
            if self.cursor <= self.limit_backward:
                return False
            self.cursor -= 1
            self.bra = self.cursor
            if not self.slice_del():
                return False
        else:
            if self.cursor != self.I_p1:
                return False
            v_3 = self.limit - self.cursor
            if not self.__r_shortv():
                return False
            self.cursor = self.limit - v_3
            c = self.cursor
            self.insert(self.cursor, self.cursor, u'e')
            self.cursor = c
    return True