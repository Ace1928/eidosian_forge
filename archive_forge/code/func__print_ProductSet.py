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
def _print_ProductSet(self, p):
    if len(p.sets) >= 1 and (not has_variety(p.sets)):
        return self._print(p.sets[0]) ** self._print(len(p.sets))
    else:
        prod_char = '×' if self._use_unicode else 'x'
        return self._print_seq(p.sets, None, None, ' %s ' % prod_char, parenthesize=lambda set: set.is_Union or set.is_Intersection or set.is_ProductSet)