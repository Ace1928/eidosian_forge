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
def _inv_transf(from_coords, to_exprs):
    inv_from = [i.as_dummy() for i in from_coords]
    inv_to = solve([t[0] - t[1] for t in zip(inv_from, to_exprs)], list(from_coords), dict=True)[0]
    inv_to = [inv_to[fc] for fc in from_coords]
    return (Matrix(inv_from), Matrix(inv_to))