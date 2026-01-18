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
@property
def _elements(self):
    """Returns all the elements of the permutation group as a list

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> p = PermutationGroup(Permutation(1, 3), Permutation(1, 2))
        >>> p._elements
        [(3), (3)(1 2), (1 3), (2 3), (1 2 3), (1 3 2)]

        """
    return list(islice(self.generate(), None))