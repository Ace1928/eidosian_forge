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
def _print_VectorZero(self, e):
    mover = self.dom.createElement('mover')
    mi = self.dom.createElement('mi')
    mi.setAttribute('mathvariant', 'bold')
    mi.appendChild(self.dom.createTextNode('0'))
    mover.appendChild(mi)
    mo = self.dom.createElement('mo')
    mo.appendChild(self.dom.createTextNode('^'))
    mover.appendChild(mo)
    return mover