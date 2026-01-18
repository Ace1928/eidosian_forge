from .basestemmer import BaseStemmer
from .among import Among
def __r_mark_yken(self):
    if not self.eq_s_b(u'ken'):
        return False
    if not self.__r_mark_suffix_with_optional_y_consonant():
        return False
    return True