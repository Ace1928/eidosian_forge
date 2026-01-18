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
def _print_nth_root(self, base, root):
    bpretty = self._print(base)
    if self._settings['use_unicode_sqrt_char'] and self._use_unicode and (root == 2) and (bpretty.height() == 1) and (bpretty.width() == 1 or (base.is_Integer and base.is_nonnegative)):
        return prettyForm(*bpretty.left('√'))
    _zZ = xobj('/', 1)
    rootsign = xobj('\\', 1) + _zZ
    rpretty = self._print(root)
    if rpretty.height() != 1:
        return self._print(base) ** self._print(1 / root)
    exp = '' if root == 2 else str(rpretty).ljust(2)
    if len(exp) > 2:
        rootsign = ' ' * (len(exp) - 2) + rootsign
    rootsign = stringPict(exp + '\n' + rootsign)
    rootsign.baseline = 0
    linelength = bpretty.height() - 1
    diagonal = stringPict('\n'.join((' ' * (linelength - i - 1) + _zZ + ' ' * i for i in range(linelength))))
    diagonal.baseline = linelength - 1
    rootsign = prettyForm(*rootsign.right(diagonal))
    rootsign.baseline = max(1, bpretty.baseline)
    s = prettyForm(hobj('_', 2 + bpretty.width()))
    s = prettyForm(*bpretty.above(s))
    s = prettyForm(*s.left(rootsign))
    return s