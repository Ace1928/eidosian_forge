from .basestemmer import BaseStemmer
from .among import Among
def __r_steps6(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_14) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.B_test1 = False
    try:
        v_1 = self.limit - self.cursor
        try:
            self.ket = self.cursor
            self.bra = self.cursor
            among_var = self.find_among_b(GreekStemmer.a_12)
            if among_var == 0:
                raise lab1()
            if self.cursor > self.limit_backward:
                raise lab1()
            if among_var == 1:
                if not self.slice_from(u'ισμ'):
                    return False
            elif not self.slice_from(u'ι'):
                return False
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        self.ket = self.cursor
        among_var = self.find_among_b(GreekStemmer.a_13)
        if among_var == 0:
            return False
        self.bra = self.cursor
        if among_var == 1:
            if not self.slice_from(u'αγνωστ'):
                return False
        elif among_var == 2:
            if not self.slice_from(u'ατομ'):
                return False
        elif among_var == 3:
            if not self.slice_from(u'γνωστ'):
                return False
        elif among_var == 4:
            if not self.slice_from(u'εθν'):
                return False
        elif among_var == 5:
            if not self.slice_from(u'εκλεκτ'):
                return False
        elif among_var == 6:
            if not self.slice_from(u'σκεπτ'):
                return False
        elif among_var == 7:
            if not self.slice_from(u'τοπ'):
                return False
        elif among_var == 8:
            if not self.slice_from(u'αλεξανδρ'):
                return False
        elif among_var == 9:
            if not self.slice_from(u'βυζαντ'):
                return False
        elif not self.slice_from(u'θεατρ'):
            return False
    except lab0:
        pass
    return True