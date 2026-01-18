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
def _needs_mul_brackets(self, expr, first=False, last=False) -> bool:
    """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of a Mul, False otherwise. This is True for Add,
        but also for some container objects that would not need brackets
        when appearing last in a Mul, e.g. an Integral. ``last=True``
        specifies that this expr is the last to appear in a Mul.
        ``first=True`` specifies that this expr is the first to appear in
        a Mul.
        """
    from sympy.concrete.products import Product
    from sympy.concrete.summations import Sum
    from sympy.integrals.integrals import Integral
    if expr.is_Mul:
        if not first and expr.could_extract_minus_sign():
            return True
    elif precedence_traditional(expr) < PRECEDENCE['Mul']:
        return True
    elif expr.is_Relational:
        return True
    if expr.is_Piecewise:
        return True
    if any((expr.has(x) for x in (Mod,))):
        return True
    if not last and any((expr.has(x) for x in (Integral, Product, Sum))):
        return True
    return False