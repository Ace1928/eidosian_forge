from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _integer_rational_reconstruction(c, m, domain):
    """
    Reconstruct a rational number `\\frac a b` from

    .. math::

        c = \\frac a b \\; \\mathrm{mod} \\, m,

    where `c` and `m` are integers.

    The algorithm is based on the Euclidean Algorithm. In general, `m` is
    not a prime number, so it is possible that `b` is not invertible modulo
    `m`. In that case ``None`` is returned.

    Parameters
    ==========

    c : Integer
        `c = \\frac a b \\; \\mathrm{mod} \\, m`
    m : Integer
        modulus, not necessarily prime
    domain : IntegerRing
        `a, b, c` are elements of ``domain``

    Returns
    =======

    frac : Rational
        either `\\frac a b` in `\\mathbb Q` or ``None``

    References
    ==========

    1. [Wang81]_

    """
    if c < 0:
        c += m
    r0, s0 = (m, domain.zero)
    r1, s1 = (c, domain.one)
    bound = sqrt(m / 2)
    while r1 >= bound:
        quo = r0 // r1
        r0, r1 = (r1, r0 - quo * r1)
        s0, s1 = (s1, s0 - quo * s1)
    if abs(s1) >= bound:
        return None
    if s1 < 0:
        a, b = (-r1, -s1)
    elif s1 > 0:
        a, b = (r1, s1)
    else:
        return None
    field = domain.get_field()
    return field(a) / field(b)