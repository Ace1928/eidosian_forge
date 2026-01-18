from math import factorial as _factorial, log, prod
from itertools import chain, islice, product
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
from sympy.combinatorics.util import (_check_cycles_alt_sym,
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq
def _coset_representative(self, g, H):
    """Return the representative of Hg from the transversal that
        would be computed by ``self.coset_transversal(H)``.

        """
    if H.order() == 1:
        return g
    if not self.base[:len(H.base)] == H.base:
        self._schreier_sims(base=H.base)
    orbits = H.basic_orbits[:]
    h_transversals = [list(_.values()) for _ in H.basic_transversals]
    transversals = [list(_.values()) for _ in self.basic_transversals]
    base = self.base
    base_ordering = _base_ordering(base, self.degree)

    def step(l, x):
        gamma = sorted(orbits[l], key=lambda y: base_ordering[y ^ x])[0]
        i = [base[l] ^ h for h in h_transversals[l]].index(gamma)
        x = h_transversals[l][i] * x
        if l < len(orbits) - 1:
            for u in transversals[l]:
                if base[l] ^ u == base[l] ^ x:
                    break
            x = step(l + 1, x * u ** (-1)) * u
        return x
    return step(0, g)