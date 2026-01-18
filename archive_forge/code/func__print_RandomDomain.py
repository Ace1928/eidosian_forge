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
def _print_RandomDomain(self, d):
    if hasattr(d, 'as_boolean'):
        pform = self._print('Domain: ')
        pform = prettyForm(*pform.right(self._print(d.as_boolean())))
        return pform
    elif hasattr(d, 'set'):
        pform = self._print('Domain: ')
        pform = prettyForm(*pform.right(self._print(d.symbols)))
        pform = prettyForm(*pform.right(self._print(' in ')))
        pform = prettyForm(*pform.right(self._print(d.set)))
        return pform
    elif hasattr(d, 'symbols'):
        pform = self._print('Domain on ')
        pform = prettyForm(*pform.right(self._print(d.symbols)))
        return pform
    else:
        return self._print(None)