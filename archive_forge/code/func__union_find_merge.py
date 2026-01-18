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
def _union_find_merge(self, first, second, ranks, parents, not_rep):
    """Merges two classes in a union-find data structure.

        Explanation
        ===========

        Used in the implementation of Atkinson's algorithm as suggested in [1],
        pp. 83-87. The class merging process uses union by rank as an
        optimization. ([7])

        Notes
        =====

        THIS FUNCTION HAS SIDE EFFECTS: the list of class representatives,
        ``parents``, the list of class sizes, ``ranks``, and the list of
        elements that are not representatives, ``not_rep``, are changed due to
        class merging.

        See Also
        ========

        minimal_block, _union_find_rep

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of computational group theory"

        .. [7] https://algorithmist.com/wiki/Union_find

        """
    rep_first = self._union_find_rep(first, parents)
    rep_second = self._union_find_rep(second, parents)
    if rep_first != rep_second:
        if ranks[rep_first] >= ranks[rep_second]:
            new_1, new_2 = (rep_first, rep_second)
        else:
            new_1, new_2 = (rep_second, rep_first)
        total_rank = ranks[new_1] + ranks[new_2]
        if total_rank > self.max_div:
            return -1
        parents[new_2] = new_1
        ranks[new_1] = total_rank
        not_rep.append(new_2)
        return 1
    return 0