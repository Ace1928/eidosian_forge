from __future__ import annotations
from typing import Any
from functools import wraps
from itertools import chain
from sympy.core import S
from sympy.core.numbers import equal_valued
from sympy.codegen.ast import (
from sympy.printing.codeprinter import CodePrinter, requires
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
from sympy.printing.codeprinter import ccode, print_ccode # noqa:F401
@wraps(meth)
def _meth_wrapper(self, expr, **kwargs):
    if expr in self.math_macros:
        return '%s%s' % (self.math_macros[expr], self._get_math_macro_suffix(real))
    else:
        return meth(self, expr, **kwargs)