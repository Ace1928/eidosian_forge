from .basestemmer import BaseStemmer
from .among import Among
def __r_Prefix_Step3a_Noun(self):
    self.bra = self.cursor
    among_var = self.find_among(ArabicStemmer.a_6)
    if among_var == 0:
        return False
    self.ket = self.cursor
    if among_var == 1:
        if not len(self.current) > 5:
            return False
        if not self.slice_del():
            return False
    else:
        if not len(self.current) > 4:
            return False
        if not self.slice_del():
            return False
    return True