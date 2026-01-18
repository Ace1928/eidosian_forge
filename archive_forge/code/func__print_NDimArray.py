import itertools
from sympy.core import S
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import Number, Rational
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from sympy.printing.conventions import requires_partial
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.str import sstr
from sympy.utilities.iterables import has_variety
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import hobj, vobj, xobj, \
def _print_NDimArray(self, expr):
    from sympy.matrices.immutable import ImmutableMatrix
    if expr.rank() == 0:
        return self._print(expr[()])
    level_str = [[]] + [[] for i in range(expr.rank())]
    shape_ranges = [list(range(i)) for i in expr.shape]
    mat = lambda x: ImmutableMatrix(x, evaluate=False)
    for outer_i in itertools.product(*shape_ranges):
        level_str[-1].append(expr[outer_i])
        even = True
        for back_outer_i in range(expr.rank() - 1, -1, -1):
            if len(level_str[back_outer_i + 1]) < expr.shape[back_outer_i]:
                break
            if even:
                level_str[back_outer_i].append(level_str[back_outer_i + 1])
            else:
                level_str[back_outer_i].append(mat(level_str[back_outer_i + 1]))
                if len(level_str[back_outer_i + 1]) == 1:
                    level_str[back_outer_i][-1] = mat([[level_str[back_outer_i][-1]]])
            even = not even
            level_str[back_outer_i + 1] = []
    out_expr = level_str[0][0]
    if expr.rank() % 2 == 1:
        out_expr = mat([out_expr])
    return self._print(out_expr)