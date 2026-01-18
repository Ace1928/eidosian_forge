from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _euclidean_algorithm(f, g, minpoly, p):
    """
    Compute the monic GCD of two univariate polynomials in
    `\\mathbb{Z}_p[z]/(\\check m_{\\alpha}(z))[x]` with the Euclidean
    Algorithm.

    In general, `\\check m_{\\alpha}(z)` is not irreducible, so it is possible
    that some leading coefficient is not invertible modulo
    `\\check m_{\\alpha}(z)`. In that case ``None`` is returned.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\\mathbb Z[x, z]`
    minpoly : PolyElement
        polynomial in `\\mathbb Z[z]`, not necessarily irreducible
    p : Integer
        prime number, modulus of `\\mathbb Z_p`

    Returns
    =======

    h : PolyElement
        GCD of `f` and `g` in `\\mathbb Z[z, x]` or ``None``, coefficients
        are in `\\left[ -\\frac{p-1} 2, \\frac{p-1} 2 \\right]`

    """
    ring = f.ring
    f = _trunc(f, minpoly, p)
    g = _trunc(g, minpoly, p)
    while g:
        rem = f
        deg = g.degree(0)
        lcinv, _, gcd = _gf_gcdex(ring.dmp_LC(g), minpoly, p)
        if not gcd == 1:
            return None
        while True:
            degrem = rem.degree(0)
            if degrem < deg:
                break
            quo = (lcinv * ring.dmp_LC(rem)).set_ring(ring)
            rem = _trunc(rem - g.mul_monom((degrem - deg, 0)) * quo, minpoly, p)
        f = g
        g = rem
    lcfinv = _gf_gcdex(ring.dmp_LC(f), minpoly, p)[0].set_ring(ring)
    return _trunc(f * lcfinv, minpoly, p)