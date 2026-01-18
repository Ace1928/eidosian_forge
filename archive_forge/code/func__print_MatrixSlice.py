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
def _print_MatrixSlice(self, m):
    from sympy.matrices import MatrixSymbol
    prettyFunc = self._print(m.parent)
    if not isinstance(m.parent, MatrixSymbol):
        prettyFunc = prettyForm(*prettyFunc.parens())

    def ppslice(x, dim):
        x = list(x)
        if x[2] == 1:
            del x[2]
        if x[0] == 0:
            x[0] = ''
        if x[1] == dim:
            x[1] = ''
        return prettyForm(*self._print_seq(x, delimiter=':'))
    prettyArgs = self._print_seq((ppslice(m.rowslice, m.parent.rows), ppslice(m.colslice, m.parent.cols)), delimiter=', ').parens(left='[', right=']')[0]
    pform = prettyForm(*stringPict.next(prettyFunc, prettyArgs), binding=prettyForm.FUNC)
    pform.prettyFunc = prettyFunc
    pform.prettyArgs = prettyArgs
    return pform