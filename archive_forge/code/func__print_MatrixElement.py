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
def _print_MatrixElement(self, expr):
    from sympy.matrices import MatrixSymbol
    if isinstance(expr.parent, MatrixSymbol) and expr.i.is_number and expr.j.is_number:
        return self._print(Symbol(expr.parent.name + '_%d%d' % (expr.i, expr.j)))
    else:
        prettyFunc = self._print(expr.parent)
        prettyFunc = prettyForm(*prettyFunc.parens())
        prettyIndices = self._print_seq((expr.i, expr.j), delimiter=', ').parens(left='[', right=']')[0]
        pform = prettyForm(*stringPict.next(prettyFunc, prettyIndices), binding=prettyForm.FUNC)
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyIndices
        return pform