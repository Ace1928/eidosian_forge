from .basestemmer import BaseStemmer
from .among import Among
def __r_VI(self):
    if not self.eq_s_b(u'i'):
        return False
    if not self.in_grouping_b(FinnishStemmer.g_V2, 97, 246):
        return False
    return True