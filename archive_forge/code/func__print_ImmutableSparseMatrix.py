from __future__ import annotations
from typing import Any
from sympy.core import Basic, Expr, Float
from sympy.core.sorting import default_sort_key
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
def _print_ImmutableSparseMatrix(self, expr):

    def print_rule(pos, val):
        return '{} -> {}'.format(self.doprint((pos[0] + 1, pos[1] + 1)), self.doprint(val))

    def print_data():
        items = sorted(expr.todok().items(), key=default_sort_key)
        return '{' + ', '.join((print_rule(k, v) for k, v in items)) + '}'

    def print_dims():
        return self.doprint(expr.shape)
    return 'SparseArray[{}, {}]'.format(print_data(), print_dims())