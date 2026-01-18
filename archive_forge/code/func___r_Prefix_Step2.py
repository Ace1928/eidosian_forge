from .basestemmer import BaseStemmer
from .among import Among
def __r_Prefix_Step2(self):
    self.bra = self.cursor
    if self.find_among(ArabicStemmer.a_5) == 0:
        return False
    self.ket = self.cursor
    if not len(self.current) > 3:
        return False
    v_1 = self.cursor
    try:
        if not self.eq_s(u'ا'):
            raise lab0()
        return False
    except lab0:
        pass
    self.cursor = v_1
    if not self.slice_del():
        return False
    return True