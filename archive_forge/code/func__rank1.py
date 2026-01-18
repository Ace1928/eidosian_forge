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
def _rank1(n, perm, inv_perm):
    if n == 1:
        return 0
    s = perm[n - 1]
    t = inv_perm[n - 1]
    perm[n - 1], perm[t] = (perm[t], s)
    inv_perm[n - 1], inv_perm[s] = (inv_perm[s], t)
    return s + n * _rank1(n - 1, perm, inv_perm)