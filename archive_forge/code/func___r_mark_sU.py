from .basestemmer import BaseStemmer
from .among import Among
def __r_mark_sU(self):
    if not self.__r_check_vowel_harmony():
        return False
    if not self.in_grouping_b(TurkishStemmer.g_U, 105, 305):
        return False
    if not self.__r_mark_suffix_with_optional_s_consonant():
        return False
    return True