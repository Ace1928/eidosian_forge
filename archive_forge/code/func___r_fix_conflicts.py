from .basestemmer import BaseStemmer
from .among import Among
def __r_fix_conflicts(self):
    self.ket = self.cursor
    among_var = self.find_among_b(LithuanianStemmer.a_2)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.slice_from(u'aitė'):
            return False
    elif among_var == 2:
        if not self.slice_from(u'uotė'):
            return False
    elif among_var == 3:
        if not self.slice_from(u'ėjimas'):
            return False
    elif among_var == 4:
        if not self.slice_from(u'esys'):
            return False
    elif among_var == 5:
        if not self.slice_from(u'asys'):
            return False
    elif among_var == 6:
        if not self.slice_from(u'avimas'):
            return False
    elif among_var == 7:
        if not self.slice_from(u'ojimas'):
            return False
    elif not self.slice_from(u'okatė'):
        return False
    return True