from .basestemmer import BaseStemmer
from .among import Among
def __r_mark_ncA(self):
    if not self.__r_check_vowel_harmony():
        return False
    if self.find_among_b(TurkishStemmer.a_11) == 0:
        return False
    if not self.__r_mark_suffix_with_optional_n_consonant():
        return False
    return True