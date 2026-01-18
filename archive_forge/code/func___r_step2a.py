from .basestemmer import BaseStemmer
from .among import Among
def __r_step2a(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_24) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    v_1 = self.limit - self.cursor
    try:
        if self.find_among_b(GreekStemmer.a_25) == 0:
            raise lab0()
        return False
    except lab0:
        pass
    self.cursor = self.limit - v_1
    c = self.cursor
    self.insert(self.cursor, self.cursor, u'αδ')
    self.cursor = c
    return True