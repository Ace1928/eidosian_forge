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
def _print_ComplexRegion(self, ts):
    if self._use_unicode:
        inn = '∊'
    else:
        inn = 'in'
    variables = self._print_seq(ts.variables)
    expr = self._print(ts.expr)
    prodsets = self._print(ts.sets)
    C = self._print_seq((variables, inn, prodsets), delimiter=' ')
    return self._hprint_vseparator(expr, C, left='{', right='}', ifascii_nougly=True, delimiter=' ')