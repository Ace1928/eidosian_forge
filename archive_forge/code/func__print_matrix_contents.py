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
def _print_matrix_contents(self, e):
    """
        This method factors out what is essentially grid printing.
        """
    M = e
    Ms = {}
    for i in range(M.rows):
        for j in range(M.cols):
            Ms[i, j] = self._print(M[i, j])
    hsep = 2
    vsep = 1
    maxw = [-1] * M.cols
    for j in range(M.cols):
        maxw[j] = max([Ms[i, j].width() for i in range(M.rows)] or [0])
    D = None
    for i in range(M.rows):
        D_row = None
        for j in range(M.cols):
            s = Ms[i, j]
            assert s.width() <= maxw[j]
            wdelta = maxw[j] - s.width()
            wleft = wdelta // 2
            wright = wdelta - wleft
            s = prettyForm(*s.right(' ' * wright))
            s = prettyForm(*s.left(' ' * wleft))
            if D_row is None:
                D_row = s
                continue
            D_row = prettyForm(*D_row.right(' ' * hsep))
            D_row = prettyForm(*D_row.right(s))
        if D is None:
            D = D_row
            continue
        for _ in range(vsep):
            D = prettyForm(*D.below(' '))
        D = prettyForm(*D.below(D_row))
    if D is None:
        D = prettyForm('')
    return D