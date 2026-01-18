from .basestemmer import BaseStemmer
from .among import Among
def __r_un_accent(self):
    v_1 = 1
    while True:
        try:
            if not self.out_grouping_b(FrenchStemmer.g_v, 97, 251):
                raise lab0()
            v_1 -= 1
            continue
        except lab0:
            pass
        break
    if v_1 > 0:
        return False
    self.ket = self.cursor
    try:
        v_3 = self.limit - self.cursor
        try:
            if not self.eq_s_b(u'é'):
                raise lab2()
            raise lab1()
        except lab2:
            pass
        self.cursor = self.limit - v_3
        if not self.eq_s_b(u'è'):
            return False
    except lab1:
        pass
    self.bra = self.cursor
    if not self.slice_from(u'e'):
        return False
    return True