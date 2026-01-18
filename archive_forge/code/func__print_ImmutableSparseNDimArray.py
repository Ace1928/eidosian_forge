from __future__ import annotations
from typing import Any
from sympy.core import Basic, Expr, Float
from sympy.core.sorting import default_sort_key
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
def _print_ImmutableSparseNDimArray(self, expr):

    def print_string_list(string_list):
        return '{' + ', '.join((a for a in string_list)) + '}'

    def to_mathematica_index(*args):
        """Helper function to change Python style indexing to
            Pathematica indexing.

            Python indexing (0, 1 ... n-1)
            -> Mathematica indexing (1, 2 ... n)
            """
        return tuple((i + 1 for i in args))

    def print_rule(pos, val):
        """Helper function to print a rule of Mathematica"""
        return '{} -> {}'.format(self.doprint(pos), self.doprint(val))

    def print_data():
        """Helper function to print data part of Mathematica
            sparse array.

            It uses the fourth notation ``SparseArray[data,{d1,d2,...}]``
            from
            https://reference.wolfram.com/language/ref/SparseArray.html

            ``data`` must be formatted with rule.
            """
        return print_string_list([print_rule(to_mathematica_index(*expr._get_tuple_index(key)), value) for key, value in sorted(expr._sparse_array.items())])

    def print_dims():
        """Helper function to print dimensions part of Mathematica
            sparse array.

            It uses the fourth notation ``SparseArray[data,{d1,d2,...}]``
            from
            https://reference.wolfram.com/language/ref/SparseArray.html
            """
        return self.doprint(expr.shape)
    return 'SparseArray[{}, {}]'.format(print_data(), print_dims())