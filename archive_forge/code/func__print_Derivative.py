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
def _print_Derivative(self, deriv):
    if requires_partial(deriv.expr) and self._use_unicode:
        deriv_symbol = U('PARTIAL DIFFERENTIAL')
    else:
        deriv_symbol = 'd'
    x = None
    count_total_deriv = 0
    for sym, num in reversed(deriv.variable_count):
        s = self._print(sym)
        ds = prettyForm(*s.left(deriv_symbol))
        count_total_deriv += num
        if not num.is_Integer or num > 1:
            ds = ds ** prettyForm(str(num))
        if x is None:
            x = ds
        else:
            x = prettyForm(*x.right(' '))
            x = prettyForm(*x.right(ds))
    f = prettyForm(*self._print(deriv.expr).parens(), binding=prettyForm.FUNC)
    pform = prettyForm(deriv_symbol)
    if (count_total_deriv > 1) != False:
        pform = pform ** prettyForm(str(count_total_deriv))
    pform = prettyForm(*pform.below(stringPict.LINE, x))
    pform.baseline = pform.baseline + 1
    pform = prettyForm(*stringPict.next(pform, f))
    pform.binding = prettyForm.MUL
    return pform