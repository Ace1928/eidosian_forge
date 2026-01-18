from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
def _print_BooleanFalse(self, expr, _type=False):
    return 'false'