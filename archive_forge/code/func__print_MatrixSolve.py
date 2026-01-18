from __future__ import annotations
from typing import Any
from sympy.core import Mul, Pow, S, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search
def _print_MatrixSolve(self, expr):
    PREC = precedence(expr)
    return '%s \\ %s' % (self.parenthesize(expr.matrix, PREC), self.parenthesize(expr.vector, PREC))