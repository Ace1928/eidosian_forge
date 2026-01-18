from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
import itertools
from sympy.core import Add, Float, Mod, Mul, Number, S, Symbol, Expr
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import Function, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true, BooleanTrue, BooleanFalse
from sympy.tensor.array import NDimArray
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE
from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str
from sympy.utilities.iterables import has_variety, sift
import re
def __print_number_polynomial(self, expr, letter, exp=None):
    if len(expr.args) == 2:
        if exp is not None:
            return '%s_{%s}^{%s}\\left(%s\\right)' % (letter, self._print(expr.args[0]), exp, self._print(expr.args[1]))
        return '%s_{%s}\\left(%s\\right)' % (letter, self._print(expr.args[0]), self._print(expr.args[1]))
    tex = '%s_{%s}' % (letter, self._print(expr.args[0]))
    if exp is not None:
        tex = '%s^{%s}' % (tex, exp)
    return tex