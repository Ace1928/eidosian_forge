from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
def _print_Indexed(self, expr):
    dims = expr.shape
    elem = S.Zero
    offset = S.One
    for i in reversed(range(expr.rank)):
        elem += expr.indices[i] * offset
        offset *= dims[i]
    return '%s[%s]' % (self._print(expr.base.label), self._print(elem))