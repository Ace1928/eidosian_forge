from __future__ import annotations
from typing import Any
from sympy.core import Basic, Expr, Float
from sympy.core.sorting import default_sort_key
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
def _print_One(self, expr):
    return '1'