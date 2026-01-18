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
def _print_GroebnerBasis(self, basis):
    exprs = [self._print_Add(arg, order=basis.order) for arg in basis.exprs]
    exprs = prettyForm(*self.join(', ', exprs).parens(left='[', right=']'))
    gens = [self._print(gen) for gen in basis.gens]
    domain = prettyForm(*prettyForm('domain=').right(self._print(basis.domain)))
    order = prettyForm(*prettyForm('order=').right(self._print(basis.order)))
    pform = self.join(', ', [exprs] + gens + [domain, order])
    pform = prettyForm(*pform.parens())
    pform = prettyForm(*pform.left(basis.__class__.__name__))
    return pform