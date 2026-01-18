from .basestemmer import BaseStemmer
from .among import Among
def __r_Step_1a(self):
    v_1 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if self.find_among_b(EnglishStemmer.a_1) == 0:
            self.cursor = self.limit - v_1
            raise lab0()
        self.bra = self.cursor
        if not self.slice_del():
            return False
    except lab0:
        pass
    self.ket = self.cursor
    among_var = self.find_among_b(EnglishStemmer.a_2)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.slice_from(u'ss'):
            return False
    elif among_var == 2:
        try:
            v_2 = self.limit - self.cursor
            try:
                c = self.cursor - 2
                if c < self.limit_backward:
                    raise lab2()
                self.cursor = c
                if not self.slice_from(u'i'):
                    return False
                raise lab1()
            except lab2:
                pass
            self.cursor = self.limit - v_2
            if not self.slice_from(u'ie'):
                return False
        except lab1:
            pass
    elif among_var == 3:
        if self.cursor <= self.limit_backward:
            return False
        self.cursor -= 1
        if not self.go_out_grouping_b(EnglishStemmer.g_v, 97, 121):
            return False
        self.cursor -= 1
        if not self.slice_del():
            return False
    return True