from .basestemmer import BaseStemmer
from .among import Among
def __r_is_reserved_word(self):
    if not self.eq_s_b(u'ad'):
        return False
    v_1 = self.limit - self.cursor
    try:
        if not self.eq_s_b(u'soy'):
            self.cursor = self.limit - v_1
            raise lab0()
    except lab0:
        pass
    if self.cursor > self.limit_backward:
        return False
    return True