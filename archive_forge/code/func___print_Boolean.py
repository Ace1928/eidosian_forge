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
def __print_Boolean(self, e, char, sort=True):
    args = e.args
    if sort:
        args = sorted(e.args, key=default_sort_key)
    arg = args[0]
    pform = self._print(arg)
    if arg.is_Boolean and (not arg.is_Not):
        pform = prettyForm(*pform.parens())
    for arg in args[1:]:
        pform_arg = self._print(arg)
        if arg.is_Boolean and (not arg.is_Not):
            pform_arg = prettyForm(*pform_arg.parens())
        pform = prettyForm(*pform.right(' %s ' % char))
        pform = prettyForm(*pform.right(pform_arg))
    return pform