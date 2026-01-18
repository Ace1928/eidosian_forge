from __future__ import annotations
from typing import Any
from functools import wraps
from sympy.core import Add, Mul, Pow, S, sympify, Float
from sympy.core.basic import Basic
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Lambda
from sympy.core.mul import _keep_coeff
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import re
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
def _get_expression_indices(self, expr, assign_to):
    from sympy.tensor import get_indices
    rinds, junk = get_indices(expr)
    linds, junk = get_indices(assign_to)
    if linds and (not rinds):
        rinds = linds
    if rinds != linds:
        raise ValueError('lhs indices must match non-dummy rhs indices in %s' % expr)
    return self._sort_optimized(rinds, assign_to)