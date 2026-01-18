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
def _print_Differential(self, diff):
    if self._use_unicode:
        d = 'ⅆ'
    else:
        d = 'd'
    field = diff._form_field
    if hasattr(field, '_coord_sys'):
        string = field._coord_sys.symbols[field._index].name
        return self._print(d + ' ' + pretty_symbol(string))
    else:
        pform = self._print(field)
        pform = prettyForm(*pform.parens())
        return prettyForm(*pform.left(d))