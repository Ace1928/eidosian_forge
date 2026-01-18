from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _evaluate_ground(f, i, a):
    """
    Evaluate a polynomial `f` at `a` in the `i`-th variable of the ground
    domain.
    """
    ring = f.ring.clone(domain=f.ring.domain.ring.drop(i))
    fa = ring.zero
    for monom, coeff in f.iterterms():
        fa[monom] = coeff.evaluate(i, a)
    return fa