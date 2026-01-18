from .basestemmer import BaseStemmer
from .among import Among
def __r_en_ending(self):
    if not self.__r_R1():
        return False
    v_1 = self.limit - self.cursor
    if not self.out_grouping_b(DutchStemmer.g_v, 97, 232):
        return False
    self.cursor = self.limit - v_1
    v_2 = self.limit - self.cursor
    try:
        if not self.eq_s_b(u'gem'):
            raise lab0()
        return False
    except lab0:
        pass
    self.cursor = self.limit - v_2
    if not self.slice_del():
        return False
    if not self.__r_undouble():
        return False
    return True