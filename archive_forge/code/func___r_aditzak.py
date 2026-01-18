from .basestemmer import BaseStemmer
from .among import Among
def __r_aditzak(self):
    self.ket = self.cursor
    among_var = self.find_among_b(BasqueStemmer.a_0)
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
        if not self.slice_from(u'atseden'):
            return False
    elif among_var == 4:
        if not self.slice_from(u'arabera'):
            return False
    elif not self.slice_from(u'baditu'):
        return False
    return True