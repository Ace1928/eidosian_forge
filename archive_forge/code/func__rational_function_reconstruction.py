from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _rational_function_reconstruction(c, p, m):
    """
    Reconstruct a rational function `\\frac a b` in `\\mathbb Z_p(t)` from

    .. math::

        c = \\frac a b \\; \\mathrm{mod} \\, m,

    where `c` and `m` are polynomials in `\\mathbb Z_p[t]` and `m` has
    positive degree.

    The algorithm is based on the Euclidean Algorithm. In general, `m` is
    not irreducible, so it is possible that `b` is not invertible modulo
    `m`. In that case ``None`` is returned.

    Parameters
    ==========

    c : PolyElement
        univariate polynomial in `\\mathbb Z[t]`
    p : Integer
        prime number
    m : PolyElement
        modulus, not necessarily irreducible

    Returns
    =======

    frac : FracElement
        either `\\frac a b` in `\\mathbb Z(t)` or ``None``

    References
    ==========

    1. [Hoeij04]_

    """
    ring = c.ring
    domain = ring.domain
    M = m.degree()
    N = M // 2
    D = M - N - 1
    r0, s0 = (m, ring.zero)
    r1, s1 = (c, ring.one)
    while r1.degree() > N:
        quo = _gf_div(r0, r1, p)[0]
        r0, r1 = (r1, (r0 - quo * r1).trunc_ground(p))
        s0, s1 = (s1, (s0 - quo * s1).trunc_ground(p))
    a, b = (r1, s1)
    if b.degree() > D or _gf_gcd(b, m, p) != 1:
        return None
    lc = b.LC
    if lc != 1:
        lcinv = domain.invert(lc, p)
        a = a.mul_ground(lcinv).trunc_ground(p)
        b = b.mul_ground(lcinv).trunc_ground(p)
    field = ring.to_field()
    return field(a) / field(b)