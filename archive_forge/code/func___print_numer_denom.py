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
def __print_numer_denom(self, p, q):
    if q == 1:
        if p < 0:
            return prettyForm(str(p), binding=prettyForm.NEG)
        else:
            return prettyForm(str(p))
    elif abs(p) >= 10 and abs(q) >= 10:
        if p < 0:
            return prettyForm(str(p), binding=prettyForm.NEG) / prettyForm(str(q))
        else:
            return prettyForm(str(p)) / prettyForm(str(q))
    else:
        return None