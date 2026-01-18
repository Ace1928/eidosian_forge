import string
from ..sage_helper import _within_sage, sage_method
class MapToGroupRingOfAbelianization(MapToAbelianization):
    """
    sage: M = Manifold('m003')
    sage: G = M.fundamental_group()
    sage: psi = MapToGroupRingOfAbelianization(G)
    sage: psi('ab') + psi('AAAAB')
    u*t + u^4*t^-4
    """

    def __init__(self, fund_group, base_ring=ZZ):
        MapToAbelianization.__init__(self, fund_group)
        self.H = H = self._range
        self.R = GroupAlgebra(H, base_ring)

    def range(self):
        return self.R

    def __call__(self, word):
        v = MapToAbelianization.__call__(self, word)
        return self.R.monomial(v)