from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
def _print_Integer(self, expr, _type=False):
    ret = super()._print_Integer(expr)
    if _type:
        return ret + '_i32'
    else:
        return ret