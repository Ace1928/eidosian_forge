from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_question_suffixes(self):
    if not self.__r_has_min_length():
        return False
    self.B_found_a_match = False
    self.limit_backward = self.cursor
    self.cursor = self.limit
    v_1 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if self.find_among_b(TamilStemmer.a_14) == 0:
            raise lab0()
        self.bra = self.cursor
        if not self.slice_from(u'்'):
            return False
        self.B_found_a_match = True
    except lab0:
        pass
    self.cursor = self.limit - v_1
    self.cursor = self.limit_backward
    self.__r_fix_endings()
    return True