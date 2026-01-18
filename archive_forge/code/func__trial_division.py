from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _trial_division(f, h, minpoly, p=None):
    """
    Check if `h` divides `f` in
    `\\mathbb K[t_1, \\ldots, t_k][z]/(m_{\\alpha}(z))`, where `\\mathbb K` is
    either `\\mathbb Q` or `\\mathbb Z_p`.

    This algorithm is based on pseudo division and does not use any
    fractions. By default `\\mathbb K` is `\\mathbb Q`, if a prime number `p`
    is given, `\\mathbb Z_p` is chosen instead.

    Parameters
    ==========

    f, h : PolyElement
        polynomials in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`
    minpoly : PolyElement
        polynomial `m_{\\alpha}(z)` in `\\mathbb Z[t_1, \\ldots, t_k][z]`
    p : Integer or None
        if `p` is given, `\\mathbb K` is set to `\\mathbb Z_p` instead of
        `\\mathbb Q`, default is ``None``

    Returns
    =======

    rem : PolyElement
        remainder of `\\frac f h`

    References
    ==========

    .. [1] [Hoeij02]_

    """
    ring = f.ring
    zxring = ring.clone(symbols=(ring.symbols[1], ring.symbols[0]))
    minpoly = minpoly.set_ring(ring)
    rem = f
    degrem = rem.degree()
    degh = h.degree()
    degm = minpoly.degree(1)
    lch = _LC(h).set_ring(ring)
    lcm = minpoly.LC
    while rem and degrem >= degh:
        lcrem = _LC(rem).set_ring(ring)
        rem = rem * lch - h.mul_monom((degrem - degh, 0)) * lcrem
        if p:
            rem = rem.trunc_ground(p)
        degrem = rem.degree(1)
        while rem and degrem >= degm:
            lcrem = _LC(rem.set_ring(zxring)).set_ring(ring)
            rem = rem.mul_ground(lcm) - minpoly.mul_monom((0, degrem - degm)) * lcrem
            if p:
                rem = rem.trunc_ground(p)
            degrem = rem.degree(1)
        degrem = rem.degree()
    return rem