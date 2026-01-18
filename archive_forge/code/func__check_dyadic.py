from sympy.core.backend import sympify, Add, ImmutableMatrix as Matrix
from sympy.core.evalf import EvalfMixin
from sympy.printing.defaults import Printable
from mpmath.libmp.libmpf import prec_to_dps
def _check_dyadic(other):
    if not isinstance(other, Dyadic):
        raise TypeError('A Dyadic must be supplied')
    return other