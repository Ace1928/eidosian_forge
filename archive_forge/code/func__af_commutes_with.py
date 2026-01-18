import random
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from sympy.core.parameters import global_parameters
from sympy.core.basic import Atom
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.sympify import _sympify
from sympy.matrices import zeros
from sympy.polys.polytools import lcm
from sympy.printing.repr import srepr
from sympy.utilities.iterables import (flatten, has_variety, minlex,
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import ifac
from sympy.multipledispatch import dispatch
def _af_commutes_with(a, b):
    """
    Checks if the two permutations with array forms
    given by ``a`` and ``b`` commute.

    Examples
    ========

    >>> from sympy.combinatorics.permutations import _af_commutes_with
    >>> _af_commutes_with([1, 2, 0], [0, 2, 1])
    False

    See Also
    ========

    Permutation, commutes_with
    """
    return not any((a[b[i]] != b[a[i]] for i in range(len(a) - 1)))