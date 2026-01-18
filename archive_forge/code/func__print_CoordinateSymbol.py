from __future__ import annotations
from typing import Any
from sympy.core.function import AppliedUndef
from sympy.core.mul import Mul
from mpmath.libmp import repr_dps, to_str as mlib_to_str
from .printer import Printer, print_function
def _print_CoordinateSymbol(self, expr):
    d = expr._assumptions.generator
    if d == {}:
        return '%s(%s, %s)' % (expr.__class__.__name__, self._print(expr.coord_sys), self._print(expr.index))
    else:
        attr = ['%s=%s' % (k, v) for k, v in d.items()]
        return '%s(%s, %s, %s)' % (expr.__class__.__name__, self._print(expr.coord_sys), self._print(expr.index), ', '.join(attr))