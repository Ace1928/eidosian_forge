from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _minpoly_from_dense(minpoly, ring):
    """
    Change representation of the minimal polynomial from ``DMP`` to
    ``PolyElement`` for a given ring.
    """
    minpoly_ = ring.zero
    for monom, coeff in minpoly.terms():
        minpoly_[monom] = ring.domain(coeff)
    return minpoly_