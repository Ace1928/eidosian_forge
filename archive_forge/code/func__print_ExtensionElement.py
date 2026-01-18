from __future__ import annotations
from typing import Any
from sympy.core.function import AppliedUndef
from sympy.core.mul import Mul
from mpmath.libmp import repr_dps, to_str as mlib_to_str
from .printer import Printer, print_function
def _print_ExtensionElement(self, f):
    rep = self._print(f.rep)
    ext = self._print(f.ext)
    return 'ExtElem(%s, %s)' % (rep, ext)