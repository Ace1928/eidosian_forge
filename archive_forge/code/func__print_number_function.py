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
def _print_number_function(self, e, name):
    pform = prettyForm(name)
    arg = self._print(e.args[0])
    pform_arg = prettyForm(' ' * arg.width())
    pform_arg = prettyForm(*pform_arg.below(arg))
    pform = prettyForm(*pform.right(pform_arg))
    if len(e.args) == 1:
        return pform
    m, x = e.args
    prettyFunc = pform
    prettyArgs = prettyForm(*self._print_seq([x]).parens())
    pform = prettyForm(*stringPict.next(prettyFunc, prettyArgs), binding=prettyForm.FUNC)
    pform.prettyFunc = prettyFunc
    pform.prettyArgs = prettyArgs
    return pform