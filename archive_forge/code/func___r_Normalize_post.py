from .basestemmer import BaseStemmer
from .among import Among
def __r_Normalize_post(self):
    v_1 = self.cursor
    try:
        self.limit_backward = self.cursor
        self.cursor = self.limit
        self.ket = self.cursor
        if self.find_among_b(ArabicStemmer.a_1) == 0:
            raise lab0()
        self.bra = self.cursor
        if not self.slice_from(u'ء'):
            return False
        self.cursor = self.limit_backward
    except lab0:
        pass
    self.cursor = v_1
    v_2 = self.cursor
    try:
        while True:
            v_3 = self.cursor
            try:
                try:
                    v_4 = self.cursor
                    try:
                        self.bra = self.cursor
                        among_var = self.find_among(ArabicStemmer.a_2)
                        if among_var == 0:
                            raise lab4()
                        self.ket = self.cursor
                        if among_var == 1:
                            if not self.slice_from(u'ا'):
                                return False
                        elif among_var == 2:
                            if not self.slice_from(u'و'):
                                return False
                        elif not self.slice_from(u'ي'):
                            return False
                        raise lab3()
                    except lab4:
                        pass
                    self.cursor = v_4
                    if self.cursor >= self.limit:
                        raise lab2()
                    self.cursor += 1
                except lab3:
                    pass
                continue
            except lab2:
                pass
            self.cursor = v_3
            break
    except lab1:
        pass
    self.cursor = v_2
    return True