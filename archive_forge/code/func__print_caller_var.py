from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
def _print_caller_var(self, expr):
    if len(expr.args) > 1:
        return '(' + self._print(expr) + ')'
    elif expr.is_number:
        return self._print(expr, _type=True)
    else:
        return self._print(expr)