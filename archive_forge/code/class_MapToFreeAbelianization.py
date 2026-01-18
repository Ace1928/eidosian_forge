import string
from ..sage_helper import _within_sage, sage_method
class MapToFreeAbelianization(MapToAbelianization):

    def range(self):
        return ZZ ** self.elementary_divisors.count(0)

    def __call__(self, word):
        D = self.elementary_divisors
        v = self.U * abelianize_word(word, self.domain_gens)
        return vector(ZZ, [v[i] for i in range(len(D)) if D[i] == 0])