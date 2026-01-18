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
def _print_Limit(self, l):
    e, z, z0, dir = l.args
    E = self._print(e)
    if precedence(e) <= PRECEDENCE['Mul']:
        E = prettyForm(*E.parens('(', ')'))
    Lim = prettyForm('lim')
    LimArg = self._print(z)
    if self._use_unicode:
        LimArg = prettyForm(*LimArg.right('─→'))
    else:
        LimArg = prettyForm(*LimArg.right('->'))
    LimArg = prettyForm(*LimArg.right(self._print(z0)))
    if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
        dir = ''
    elif self._use_unicode:
        dir = '⁺' if str(dir) == '+' else '⁻'
    LimArg = prettyForm(*LimArg.right(self._print(dir)))
    Lim = prettyForm(*Lim.below(LimArg))
    Lim = prettyForm(*Lim.right(E), binding=prettyForm.MUL)
    return Lim