from __future__ import annotations
from typing import Any
from functools import reduce
from itertools import permutations
from sympy.combinatorics import Permutation
from sympy.core import (
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol, Dummy
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions import factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix
from sympy.solvers import solve
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.simplify.simplify import simplify
@staticmethod
def _solve_inverse(sym1, sym2, exprs, sys1_name, sys2_name):
    ret = solve([t[0] - t[1] for t in zip(sym2, exprs)], list(sym1), dict=True)
    if len(ret) == 0:
        temp = 'Cannot solve inverse relation from {} to {}.'
        raise NotImplementedError(temp.format(sys1_name, sys2_name))
    elif len(ret) > 1:
        temp = 'Obtained multiple inverse relation from {} to {}.'
        raise ValueError(temp.format(sys1_name, sys2_name))
    return ret[0]