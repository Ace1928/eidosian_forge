from .basestemmer import BaseStemmer
from .among import Among
def __r_tidy_up(self):
    self.ket = self.cursor
    among_var = self.find_among_b(RussianStemmer.a_7)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.slice_del():
            return False
        self.ket = self.cursor
        if not self.eq_s_b(u'н'):
            return False
        self.bra = self.cursor
        if not self.eq_s_b(u'н'):
            return False
        if not self.slice_del():
            return False
    elif among_var == 2:
        if not self.eq_s_b(u'н'):
            return False
        if not self.slice_del():
            return False
    elif not self.slice_del():
        return False
    return True