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
@classmethod
def _inverse_transformation(cls, sys1, sys2):
    forward = sys1.transform(sys2)
    inv_results = cls._solve_inverse(sys1.symbols, sys2.symbols, forward, sys1.name, sys2.name)
    signature = tuple(sys1.symbols)
    return [inv_results[s] for s in signature]