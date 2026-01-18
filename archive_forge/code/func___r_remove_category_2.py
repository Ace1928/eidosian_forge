from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_category_2(self):
    self.ket = self.cursor
    among_var = self.find_among_b(NepaliStemmer.a_2)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        try:
            v_1 = self.limit - self.cursor
            try:
                if not self.eq_s_b(u'यौ'):
                    raise lab1()
                raise lab0()
            except lab1:
                pass
            self.cursor = self.limit - v_1
            try:
                if not self.eq_s_b(u'छौ'):
                    raise lab2()
                raise lab0()
            except lab2:
                pass
            self.cursor = self.limit - v_1
            try:
                if not self.eq_s_b(u'नौ'):
                    raise lab3()
                raise lab0()
            except lab3:
                pass
            self.cursor = self.limit - v_1
            if not self.eq_s_b(u'थे'):
                return False
        except lab0:
            pass
        if not self.slice_del():
            return False
    else:
        if not self.eq_s_b(u'त्र'):
            return False
        if not self.slice_del():
            return False
    return True