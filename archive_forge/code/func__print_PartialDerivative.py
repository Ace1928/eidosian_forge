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
def _print_PartialDerivative(self, deriv):
    if self._use_unicode:
        deriv_symbol = U('PARTIAL DIFFERENTIAL')
    else:
        deriv_symbol = 'd'
    x = None
    for variable in reversed(deriv.variables):
        s = self._print(variable)
        ds = prettyForm(*s.left(deriv_symbol))
        if x is None:
            x = ds
        else:
            x = prettyForm(*x.right(' '))
            x = prettyForm(*x.right(ds))
    f = prettyForm(*self._print(deriv.expr).parens(), binding=prettyForm.FUNC)
    pform = prettyForm(deriv_symbol)
    if len(deriv.variables) > 1:
        pform = pform ** self._print(len(deriv.variables))
    pform = prettyForm(*pform.below(stringPict.LINE, x))
    pform.baseline = pform.baseline + 1
    pform = prettyForm(*stringPict.next(pform, f))
    pform.binding = prettyForm.MUL
    return pform