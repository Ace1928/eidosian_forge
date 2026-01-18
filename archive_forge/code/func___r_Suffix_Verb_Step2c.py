from .basestemmer import BaseStemmer
from .among import Among
def __r_Suffix_Verb_Step2c(self):
    self.ket = self.cursor
    among_var = self.find_among_b(ArabicStemmer.a_20)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not len(self.current) >= 4:
            return False
        if not self.slice_del():
            return False
    else:
        if not len(self.current) >= 6:
            return False
        if not self.slice_del():
            return False
    return True