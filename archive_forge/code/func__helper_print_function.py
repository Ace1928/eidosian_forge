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
def _helper_print_function(self, func, args, sort=False, func_name=None, delimiter=', ', elementwise=False, left='(', right=')'):
    if sort:
        args = sorted(args, key=default_sort_key)
    if not func_name and hasattr(func, '__name__'):
        func_name = func.__name__
    if func_name:
        prettyFunc = self._print(Symbol(func_name))
    else:
        prettyFunc = prettyForm(*self._print(func).parens())
    if elementwise:
        if self._use_unicode:
            circ = pretty_atom('Modifier Letter Low Ring')
        else:
            circ = '.'
        circ = self._print(circ)
        prettyFunc = prettyForm(*stringPict.next(prettyFunc, circ), binding=prettyForm.LINE)
    prettyArgs = prettyForm(*self._print_seq(args, delimiter=delimiter).parens(left=left, right=right))
    pform = prettyForm(*stringPict.next(prettyFunc, prettyArgs), binding=prettyForm.FUNC)
    pform.prettyFunc = prettyFunc
    pform.prettyArgs = prettyArgs
    return pform