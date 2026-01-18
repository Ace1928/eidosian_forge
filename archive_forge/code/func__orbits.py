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
def _orbits(degree, generators):
    """Compute the orbits of G.

    If ``rep=False`` it returns a list of sets else it returns a list of
    representatives of the orbits

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.perm_groups import _orbits
    >>> a = Permutation([0, 2, 1])
    >>> b = Permutation([1, 0, 2])
    >>> _orbits(a.size, [a, b])
    [{0, 1, 2}]
    """
    orbs = []
    sorted_I = list(range(degree))
    I = set(sorted_I)
    while I:
        i = sorted_I[0]
        orb = _orbit(degree, generators, i)
        orbs.append(orb)
        I -= orb
        sorted_I = [i for i in sorted_I if i not in orb]
    return orbs