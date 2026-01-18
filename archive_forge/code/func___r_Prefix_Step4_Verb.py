from .basestemmer import BaseStemmer
from .among import Among
def __r_Prefix_Step4_Verb(self):
    self.bra = self.cursor
    if self.find_among(ArabicStemmer.a_9) == 0:
        return False
    self.ket = self.cursor
    if not len(self.current) > 4:
        return False
    self.B_is_verb = True
    self.B_is_noun = False
    if not self.slice_from(u'است'):
        return False
    return True