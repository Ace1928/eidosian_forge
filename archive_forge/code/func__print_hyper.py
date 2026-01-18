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
def _print_hyper(self, e):
    ap = [self._print(a) for a in e.ap]
    bq = [self._print(b) for b in e.bq]
    P = self._print(e.argument)
    P.baseline = P.height() // 2
    D = None
    for v in [ap, bq]:
        D_row = self._hprint_vec(v)
        if D is None:
            D = D_row
        else:
            D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
    D.baseline = D.height() // 2
    P = prettyForm(*P.left(' '))
    D = prettyForm(*D.right(' '))
    D = self._hprint_vseparator(D, P)
    D = prettyForm(*D.parens('(', ')'))
    above = D.height() // 2 - 1
    below = D.height() - above - 1
    sz, t, b, add, img = annotated('F')
    F = prettyForm('\n' * (above - t) + img + '\n' * (below - b), baseline=above + sz)
    add = (sz + 1) // 2
    F = prettyForm(*F.left(self._print(len(e.ap))))
    F = prettyForm(*F.right(self._print(len(e.bq))))
    F.baseline = above + add
    D = prettyForm(*F.right(' ', D))
    return D