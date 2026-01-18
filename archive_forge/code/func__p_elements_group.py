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
def _p_elements_group(self, p):
    """
        For an abelian p-group, return the subgroup consisting of
        all elements of order p (and the identity)

        """
    gens = self.generators[:]
    gens = sorted(gens, key=lambda x: x.order(), reverse=True)
    gens_p = [g ** (g.order() / p) for g in gens]
    gens_r = []
    for i in range(len(gens)):
        x = gens[i]
        x_order = x.order()
        x_p = x ** (x_order / p)
        if i > 0:
            P = PermutationGroup(gens_p[:i])
        else:
            P = PermutationGroup(self.identity)
        if x ** (x_order / p) not in P:
            gens_r.append(x ** (x_order / p))
        else:
            g = P.generator_product(x_p, original=True)
            for s in g:
                x = x * s ** (-1)
            x_order = x_order / p
            del gens[i]
            del gens_p[i]
            j = i - 1
            while j < len(gens) and gens[j].order() >= x_order:
                j += 1
            gens = gens[:j] + [x] + gens[j:]
            gens_p = gens_p[:j] + [x] + gens_p[j:]
    return PermutationGroup(gens_r)