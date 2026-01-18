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
def _eval_is_alt_sym_monte_carlo(self, eps=0.05, perms=None):
    """A test using monte-carlo algorithm.

        Parameters
        ==========

        eps : float, optional
            The criterion for the incorrect ``False`` return.

        perms : list[Permutation], optional
            If explicitly given, it tests over the given candidates
            for testing.

            If ``None``, it randomly computes ``N_eps`` and chooses
            ``N_eps`` sample of the permutation from the group.

        See Also
        ========

        _check_cycles_alt_sym
        """
    if perms is None:
        n = self.degree
        if n < 17:
            c_n = 0.34
        else:
            c_n = 0.57
        d_n = c_n * log(2) / log(n)
        N_eps = int(-log(eps) / d_n)
        perms = (self.random_pr() for i in range(N_eps))
        return self._eval_is_alt_sym_monte_carlo(perms=perms)
    for perm in perms:
        if _check_cycles_alt_sym(perm):
            return True
    return False