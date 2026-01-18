from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _gf_gcd(fp, gp, p):
    """
    Compute the GCD of two univariate polynomials in `\\mathbb{Z}_p[x]`.
    """
    dom = fp.ring.domain
    while gp:
        rem = fp
        deg = gp.degree()
        lcinv = dom.invert(gp.LC, p)
        while True:
            degrem = rem.degree()
            if degrem < deg:
                break
            rem = (rem - gp.mul_monom((degrem - deg,)).mul_ground(lcinv * rem.LC)).trunc_ground(p)
        fp = gp
        gp = rem
    return fp.mul_ground(dom.invert(fp.LC, p)).trunc_ground(p)