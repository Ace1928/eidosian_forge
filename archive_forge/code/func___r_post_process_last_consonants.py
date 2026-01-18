from .basestemmer import BaseStemmer
from .among import Among
def __r_post_process_last_consonants(self):
    self.ket = self.cursor
    among_var = self.find_among_b(TurkishStemmer.a_23)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.slice_from(u'p'):
            return False
    elif among_var == 2:
        if not self.slice_from(u'ç'):
            return False
    elif among_var == 3:
        if not self.slice_from(u't'):
            return False
    elif not self.slice_from(u'k'):
        return False
    return True