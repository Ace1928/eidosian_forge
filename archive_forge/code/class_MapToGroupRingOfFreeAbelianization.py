import string
from ..sage_helper import _within_sage, sage_method
class MapToGroupRingOfFreeAbelianization(MapToFreeAbelianization):

    def __init__(self, fund_group, base_ring=ZZ):
        MapToFreeAbelianization.__init__(self, fund_group)
        n = self.elementary_divisors.count(0)
        self.R = LaurentPolynomialRing(base_ring, list(string.ascii_lowercase[:n]))

    def range(self):
        return self.R

    def __call__(self, word):
        v = MapToFreeAbelianization.__call__(self, word)
        return prod([g ** v[i] for i, g in enumerate(self.R.gens())])