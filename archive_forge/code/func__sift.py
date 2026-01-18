from sympy.ntheory.primetest import isprime
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.printing.defaults import DefaultPrinting
from sympy.combinatorics.free_groups import free_group
def _sift(self, z, g):
    h = g
    d = self.depth(h)
    while d < len(self.pcgs) and z[d - 1] != 1:
        k = z[d - 1]
        e = self.leading_exponent(h) * self.leading_exponent(k) ** (-1)
        e = e % self.relative_order[d - 1]
        h = k ** (-e) * h
        d = self.depth(h)
    return h