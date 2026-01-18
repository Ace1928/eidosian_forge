from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
def _traverse_matrix_indices(self, mat):
    rows, cols = mat.shape
    return ((i, j) for i in range(rows) for j in range(cols))