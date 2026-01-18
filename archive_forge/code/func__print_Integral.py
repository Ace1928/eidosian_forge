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
def _print_Integral(self, integral):
    f = integral.function
    prettyF = self._print(f)
    if f.is_Add:
        prettyF = prettyForm(*prettyF.parens())
    arg = prettyF
    for x in integral.limits:
        prettyArg = self._print(x[0])
        if prettyArg.width() > 1:
            prettyArg = prettyForm(*prettyArg.parens())
        arg = prettyForm(*arg.right(' d', prettyArg))
    firstterm = True
    s = None
    for lim in integral.limits:
        h = arg.height()
        H = h + 2
        ascii_mode = not self._use_unicode
        if ascii_mode:
            H += 2
        vint = vobj('int', H)
        pform = prettyForm(vint)
        pform.baseline = arg.baseline + (H - h) // 2
        if len(lim) > 1:
            if len(lim) == 2:
                prettyA = prettyForm('')
                prettyB = self._print(lim[1])
            if len(lim) == 3:
                prettyA = self._print(lim[1])
                prettyB = self._print(lim[2])
            if ascii_mode:
                spc = max(1, 3 - prettyB.width())
                prettyB = prettyForm(*prettyB.left(' ' * spc))
                spc = max(1, 4 - prettyA.width())
                prettyA = prettyForm(*prettyA.right(' ' * spc))
            pform = prettyForm(*pform.above(prettyB))
            pform = prettyForm(*pform.below(prettyA))
        if not ascii_mode:
            pform = prettyForm(*pform.right(' '))
        if firstterm:
            s = pform
            firstterm = False
        else:
            s = prettyForm(*s.left(pform))
    pform = prettyForm(*arg.left(s))
    pform.binding = prettyForm.MUL
    return pform