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
def _print_SetOp(self, expr, symbol, prec):
    mrow = self.dom.createElement('mrow')
    mrow.appendChild(self.parenthesize(expr.args[0], prec))
    for arg in expr.args[1:]:
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode(symbol))
        y = self.parenthesize(arg, prec)
        mrow.appendChild(x)
        mrow.appendChild(y)
    return mrow