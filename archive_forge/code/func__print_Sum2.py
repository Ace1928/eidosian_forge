from __future__ import annotations
from typing import Any
from sympy.core.function import AppliedUndef
from sympy.core.mul import Mul
from mpmath.libmp import repr_dps, to_str as mlib_to_str
from .printer import Printer, print_function
def _print_Sum2(self, expr):
    return 'Sum2(%s, (%s, %s, %s))' % (self._print(expr.f), self._print(expr.i), self._print(expr.a), self._print(expr.b))