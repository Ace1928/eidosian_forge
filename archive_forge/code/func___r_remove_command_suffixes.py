from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_command_suffixes(self):
    if not self.__r_has_min_length():
        return False
    self.B_found_a_match = False
    self.limit_backward = self.cursor
    self.cursor = self.limit
    self.ket = self.cursor
    if self.find_among_b(TamilStemmer.a_15) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.B_found_a_match = True
    self.cursor = self.limit_backward
    return True