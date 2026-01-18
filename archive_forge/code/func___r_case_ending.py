from .basestemmer import BaseStemmer
from .among import Among
def __r_case_ending(self):
    if self.cursor < self.I_p1:
        return False
    v_2 = self.limit_backward
    self.limit_backward = self.I_p1
    self.ket = self.cursor
    among_var = self.find_among_b(FinnishStemmer.a_6)
    if among_var == 0:
        self.limit_backward = v_2
        return False
    self.bra = self.cursor
    self.limit_backward = v_2
    if among_var == 1:
        if not self.eq_s_b(u'a'):
            return False
    elif among_var == 2:
        if not self.eq_s_b(u'e'):
            return False
    elif among_var == 3:
        if not self.eq_s_b(u'i'):
            return False
    elif among_var == 4:
        if not self.eq_s_b(u'o'):
            return False
    elif among_var == 5:
        if not self.eq_s_b(u'ä'):
            return False
    elif among_var == 6:
        if not self.eq_s_b(u'ö'):
            return False
    elif among_var == 7:
        v_3 = self.limit - self.cursor
        try:
            v_4 = self.limit - self.cursor
            try:
                v_5 = self.limit - self.cursor
                try:
                    if not self.__r_LONG():
                        raise lab2()
                    raise lab1()
                except lab2:
                    pass
                self.cursor = self.limit - v_5
                if not self.eq_s_b(u'ie'):
                    self.cursor = self.limit - v_3
                    raise lab0()
            except lab1:
                pass
            self.cursor = self.limit - v_4
            if self.cursor <= self.limit_backward:
                self.cursor = self.limit - v_3
                raise lab0()
            self.cursor -= 1
            self.bra = self.cursor
        except lab0:
            pass
    elif among_var == 8:
        if not self.in_grouping_b(FinnishStemmer.g_V1, 97, 246):
            return False
        if not self.in_grouping_b(FinnishStemmer.g_C, 98, 122):
            return False
    if not self.slice_del():
        return False
    self.B_ending_removed = True
    return True