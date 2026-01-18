from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _gf_div(f, g, p):
    """
    Compute `\\frac f g` modulo `p` for two univariate polynomials over
    `\\mathbb Z_p`.
    """
    ring = f.ring
    densequo, denserem = gf_div(f.to_dense(), g.to_dense(), p, ring.domain)
    return (ring.from_dense(densequo), ring.from_dense(denserem))