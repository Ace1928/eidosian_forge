from .basestemmer import BaseStemmer
from .among import Among
def __r_mark_ysA(self):
    if self.find_among_b(TurkishStemmer.a_21) == 0:
        return False
    if not self.__r_mark_suffix_with_optional_y_consonant():
        return False
    return True