from .basestemmer import BaseStemmer
from .among import Among
def __r_mark_regions(self):
    self.I_p1 = self.limit
    self.I_p2 = self.limit
    v_1 = self.cursor
    try:
        if not self.go_out_grouping(CatalanStemmer.g_v, 97, 252):
            raise lab0()
        self.cursor += 1
        if not self.go_in_grouping(CatalanStemmer.g_v, 97, 252):
            raise lab0()
        self.cursor += 1
        self.I_p1 = self.cursor
        if not self.go_out_grouping(CatalanStemmer.g_v, 97, 252):
            raise lab0()
        self.cursor += 1
        if not self.go_in_grouping(CatalanStemmer.g_v, 97, 252):
            raise lab0()
        self.cursor += 1
        self.I_p2 = self.cursor
    except lab0:
        pass
    self.cursor = v_1
    return True