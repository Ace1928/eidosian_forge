from .basestemmer import BaseStemmer
from .among import Among
def __r_particle_etc(self):
    if self.cursor < self.I_p1:
        return False
    v_2 = self.limit_backward
    self.limit_backward = self.I_p1
    self.ket = self.cursor
    among_var = self.find_among_b(FinnishStemmer.a_0)
    if among_var == 0:
        self.limit_backward = v_2
        return False
    self.bra = self.cursor
    self.limit_backward = v_2
    if among_var == 1:
        if not self.in_grouping_b(FinnishStemmer.g_particle_end, 97, 246):
            return False
    elif not self.__r_R2():
        return False
    if not self.slice_del():
        return False
    return True