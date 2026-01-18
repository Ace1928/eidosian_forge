from .basestemmer import BaseStemmer
from .among import Among
def __r_factive(self):
    self.ket = self.cursor
    if self.find_among_b(HungarianStemmer.a_7) == 0:
        return False
    self.bra = self.cursor
    if not self.__r_R1():
        return False
    if not self.__r_double():
        return False
    if not self.slice_del():
        return False
    if not self.__r_undouble():
        return False
    return True