from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
def _not_a_coeff(expr):
    """Do not treat NaN and infinities as valid polynomial coefficients. """
    if type(expr) in illegal_types or expr in finf:
        return True
    if isinstance(expr, float) and float(expr) != expr:
        return True
    return