from .basestemmer import BaseStemmer
from .among import Among
def __r_initial_morph(self):
    self.bra = self.cursor
    among_var = self.find_among(IrishStemmer.a_0)
    if among_var == 0:
        return False
    self.ket = self.cursor
    if among_var == 1:
        if not self.slice_del():
            return False
    elif among_var == 2:
        if not self.slice_from(u'f'):
            return False
    elif among_var == 3:
        if not self.slice_from(u's'):
            return False
    elif among_var == 4:
        if not self.slice_from(u'b'):
            return False
    elif among_var == 5:
        if not self.slice_from(u'c'):
            return False
    elif among_var == 6:
        if not self.slice_from(u'd'):
            return False
    elif among_var == 7:
        if not self.slice_from(u'g'):
            return False
    elif among_var == 8:
        if not self.slice_from(u'p'):
            return False
    elif among_var == 9:
        if not self.slice_from(u't'):
            return False
    elif not self.slice_from(u'm'):
        return False
    return True