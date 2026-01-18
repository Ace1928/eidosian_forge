from .basestemmer import BaseStemmer
from .among import Among
def __r_Prefix_Step3_Verb(self):
    self.bra = self.cursor
    among_var = self.find_among(ArabicStemmer.a_8)
    if among_var == 0:
        return False
    self.ket = self.cursor
    if among_var == 1:
        if not len(self.current) > 4:
            return False
        if not self.slice_from(u'ي'):
            return False
    elif among_var == 2:
        if not len(self.current) > 4:
            return False
        if not self.slice_from(u'ت'):
            return False
    elif among_var == 3:
        if not len(self.current) > 4:
            return False
        if not self.slice_from(u'ن'):
            return False
    else:
        if not len(self.current) > 4:
            return False
        if not self.slice_from(u'أ'):
            return False
    return True