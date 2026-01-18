from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_pronoun_prefixes(self):
    self.B_found_a_match = False
    self.bra = self.cursor
    if self.find_among(TamilStemmer.a_11) == 0:
        return False
    if self.find_among(TamilStemmer.a_12) == 0:
        return False
    if not self.eq_s(u'்'):
        return False
    self.ket = self.cursor
    if not self.slice_del():
        return False
    self.B_found_a_match = True
    v_1 = self.cursor
    self.__r_fix_va_start()
    self.cursor = v_1
    return True