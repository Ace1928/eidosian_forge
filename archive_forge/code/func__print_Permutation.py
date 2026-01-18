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
def _print_Permutation(self, expr):
    from sympy.combinatorics.permutations import Permutation, Cycle
    perm_cyclic = Permutation.print_cyclic
    if perm_cyclic is not None:
        sympy_deprecation_warning(f'\n                Setting Permutation.print_cyclic is deprecated. Instead use\n                init_printing(perm_cyclic={perm_cyclic}).\n                ', deprecated_since_version='1.6', active_deprecations_target='deprecated-permutation-print_cyclic', stacklevel=7)
    else:
        perm_cyclic = self._settings.get('perm_cyclic', True)
    if perm_cyclic:
        return self._print_Cycle(Cycle(expr))
    lower = expr.array_form
    upper = list(range(len(lower)))
    result = stringPict('')
    first = True
    for u, l in zip(upper, lower):
        s1 = self._print(u)
        s2 = self._print(l)
        col = prettyForm(*s1.below(s2))
        if first:
            first = False
        else:
            col = prettyForm(*col.left(' '))
        result = prettyForm(*result.right(col))
    return prettyForm(*result.parens())