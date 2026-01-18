from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_ArrayDiagonal(self, expr):
    from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
    diagonal_indices = list(expr.diagonal_indices)
    if isinstance(expr.expr, ArrayTensorProduct):
        subranks = expr.expr.subranks
        elems = expr.expr.args
    else:
        subranks = expr.subranks
        elems = [expr.expr]
    diagonal_string, letters_free, letters_dum = self._get_einsum_string(subranks, diagonal_indices)
    elems = [self._print(i) for i in elems]
    return '%s("%s", %s)' % (self._module_format(self._module + '.' + self._einsum), '{}->{}'.format(diagonal_string, ''.join(letters_free + letters_dum)), ', '.join(elems))