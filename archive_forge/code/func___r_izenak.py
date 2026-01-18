from .basestemmer import BaseStemmer
from .among import Among
def __r_izenak(self):
    self.ket = self.cursor
    among_var = self.find_among_b(BasqueStemmer.a_1)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.__r_RV():
            return False
        if not self.slice_del():
            return False
    elif among_var == 2:
        if not self.__r_R2():
            return False
        if not self.slice_del():
            return False
    elif among_var == 3:
        if not self.slice_from(u'jok'):
            return False
    elif among_var == 4:
        if not self.__r_R1():
            return False
        if not self.slice_del():
            return False
    elif among_var == 5:
        if not self.slice_from(u'tra'):
            return False
    elif among_var == 6:
        if not self.slice_from(u'minutu'):
            return False
    elif among_var == 7:
        if not self.slice_from(u'zehar'):
            return False
    elif among_var == 8:
        if not self.slice_from(u'geldi'):
            return False
    elif among_var == 9:
        if not self.slice_from(u'igaro'):
            return False
    elif not self.slice_from(u'aurka'):
        return False
    return True