from sympy.polys.domains import QQ, EX
from sympy.polys.rings import PolyElement, ring, sring
from sympy.polys.polyerrors import DomainError
from sympy.polys.monomials import (monomial_min, monomial_mul, monomial_div,
from mpmath.libmp.libintmath import ifac
from sympy.core import PoleError, Function, Expr
from sympy.core.numbers import Rational, igcd
from sympy.functions import sin, cos, tan, atan, exp, atanh, tanh, log, ceiling
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import giant_steps
import math
def _check_series_var(p, x, name):
    index = p.ring.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    if m < 0:
        raise PoleError('Asymptotic expansion of %s around [oo] not implemented.' % name)
    return (index, m)