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
def _print_Subs(self, e):
    pform = self._print(e.expr)
    pform = prettyForm(*pform.parens())
    h = pform.height() if pform.height() > 1 else 2
    rvert = stringPict(vobj('|', h), baseline=pform.baseline)
    pform = prettyForm(*pform.right(rvert))
    b = pform.baseline
    pform.baseline = pform.height() - 1
    pform = prettyForm(*pform.right(self._print_seq([self._print_seq((self._print(v[0]), xsym('=='), self._print(v[1])), delimiter='') for v in zip(e.variables, e.point)])))
    pform.baseline = b
    return pform