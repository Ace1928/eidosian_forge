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
def _print_Diagram(self, diagram):
    if not diagram.premises:
        return self._print(S.EmptySet)
    pretty_result = self._print(diagram.premises)
    if diagram.conclusions:
        results_arrow = ' %s ' % xsym('==>')
        pretty_conclusions = self._print(diagram.conclusions)[0]
        pretty_result = pretty_result.right(results_arrow, pretty_conclusions)
    return prettyForm(pretty_result[0])