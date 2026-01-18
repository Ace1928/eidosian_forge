from .basestemmer import BaseStemmer
from .among import Among
def __r_adjetiboak(self):
    self.ket = self.cursor
    among_var = self.find_among_b(BasqueStemmer.a_2)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.__r_RV():
            return False
        if not self.slice_del():
            return False
    elif not self.slice_from(u'z'):
        return False
    return True