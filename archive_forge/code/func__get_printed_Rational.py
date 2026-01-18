from __future__ import annotations
from typing import Any
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import \
from sympy.printing.pretty.pretty_symbology import greek_unicode
from sympy.printing.printer import Printer, print_function
from mpmath.libmp import prec_to_dps, repr_dps, to_str as mlib_to_str
def _get_printed_Rational(self, e, folded=None):
    if e.p < 0:
        p = -e.p
    else:
        p = e.p
    x = self.dom.createElement('mfrac')
    if folded or self._settings['fold_short_frac']:
        x.setAttribute('bevelled', 'true')
    x.appendChild(self._print(p))
    x.appendChild(self._print(e.q))
    if e.p < 0:
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('-'))
        mrow.appendChild(mo)
        mrow.appendChild(x)
        return mrow
    else:
        return x