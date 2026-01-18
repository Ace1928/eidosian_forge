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
def _union_find_rep(self, num, parents):
    """Find representative of a class in a union-find data structure.

        Explanation
        ===========

        Used in the implementation of Atkinson's algorithm as suggested in [1],
        pp. 83-87. After the representative of the class to which ``num``
        belongs is found, path compression is performed as an optimization
        ([7]).

        Notes
        =====

        THIS FUNCTION HAS SIDE EFFECTS: the list of class representatives,
        ``parents``, is altered due to path compression.

        See Also
        ========

        minimal_block, _union_find_merge

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of computational group theory"

        .. [7] https://algorithmist.com/wiki/Union_find

        """
    rep, parent = (num, parents[num])
    while parent != rep:
        rep = parent
        parent = parents[rep]
    temp, parent = (num, parents[num])
    while parent != rep:
        parents[temp] = rep
        temp = parent
        parent = parents[temp]
    return rep