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
def _print_Feedback(self, expr):
    from sympy.physics.control import TransferFunction, Series
    num, tf = (expr.sys1, TransferFunction(1, 1, expr.var))
    num_arg_list = list(num.args) if isinstance(num, Series) else [num]
    den_arg_list = list(expr.sys2.args) if isinstance(expr.sys2, Series) else [expr.sys2]
    if isinstance(num, Series) and isinstance(expr.sys2, Series):
        den = Series(*num_arg_list, *den_arg_list)
    elif isinstance(num, Series) and isinstance(expr.sys2, TransferFunction):
        if expr.sys2 == tf:
            den = Series(*num_arg_list)
        else:
            den = Series(*num_arg_list, expr.sys2)
    elif isinstance(num, TransferFunction) and isinstance(expr.sys2, Series):
        if num == tf:
            den = Series(*den_arg_list)
        else:
            den = Series(num, *den_arg_list)
    elif num == tf:
        den = Series(*den_arg_list)
    elif expr.sys2 == tf:
        den = Series(*num_arg_list)
    else:
        den = Series(*num_arg_list, *den_arg_list)
    denom = prettyForm(*stringPict.next(self._print(tf)))
    denom.baseline = denom.height() // 2
    denom = prettyForm(*stringPict.next(denom, ' + ')) if expr.sign == -1 else prettyForm(*stringPict.next(denom, ' - '))
    denom = prettyForm(*stringPict.next(denom, self._print(den)))
    return self._print(num) / denom