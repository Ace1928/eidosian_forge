from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _chinese_remainder_reconstruction_multivariate(hp, hq, p, q):
    """
    Construct a polynomial `h_{pq}` in
    `\\mathbb{Z}_{p q}[x_0, \\ldots, x_{k-1}]` such that

    .. math ::

        h_{pq} = h_p \\; \\mathrm{mod} \\, p

        h_{pq} = h_q \\; \\mathrm{mod} \\, q

    for relatively prime integers `p` and `q` and polynomials
    `h_p` and `h_q` in `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]` and
    `\\mathbb{Z}_q[x_0, \\ldots, x_{k-1}]` respectively.

    The coefficients of the polynomial `h_{pq}` are computed with the
    Chinese Remainder Theorem. The symmetric representation in
    `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`,
    `\\mathbb{Z}_q[x_0, \\ldots, x_{k-1}]` and
    `\\mathbb{Z}_{p q}[x_0, \\ldots, x_{k-1}]` is used.

    Parameters
    ==========

    hp : PolyElement
        multivariate integer polynomial with coefficients in `\\mathbb{Z}_p`
    hq : PolyElement
        multivariate integer polynomial with coefficients in `\\mathbb{Z}_q`
    p : Integer
        modulus of `h_p`, relatively prime to `q`
    q : Integer
        modulus of `h_q`, relatively prime to `p`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _chinese_remainder_reconstruction_multivariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)
    >>> p = 3
    >>> q = 5

    >>> hp = x**3*y - x**2 - 1
    >>> hq = -x**3*y - 2*x*y**2 + 2

    >>> hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)
    >>> hpq
    4*x**3*y + 5*x**2 + 3*x*y**2 + 2

    >>> hpq.trunc_ground(p) == hp
    True
    >>> hpq.trunc_ground(q) == hq
    True

    >>> R, x, y, z = ring("x, y, z", ZZ)
    >>> p = 6
    >>> q = 5

    >>> hp = 3*x**4 - y**3*z + z
    >>> hq = -2*x**4 + z

    >>> hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)
    >>> hpq
    3*x**4 + 5*y**3*z + z

    >>> hpq.trunc_ground(p) == hp
    True
    >>> hpq.trunc_ground(q) == hq
    True

    """
    hpmonoms = set(hp.monoms())
    hqmonoms = set(hq.monoms())
    monoms = hpmonoms.intersection(hqmonoms)
    hpmonoms.difference_update(monoms)
    hqmonoms.difference_update(monoms)
    zero = hp.ring.domain.zero
    hpq = hp.ring.zero
    if isinstance(hp.ring.domain, PolynomialRing):
        crt_ = _chinese_remainder_reconstruction_multivariate
    else:

        def crt_(cp, cq, p, q):
            return crt([p, q], [cp, cq], symmetric=True)[0]
    for monom in monoms:
        hpq[monom] = crt_(hp[monom], hq[monom], p, q)
    for monom in hpmonoms:
        hpq[monom] = crt_(hp[monom], zero, p, q)
    for monom in hqmonoms:
        hpq[monom] = crt_(zero, hq[monom], p, q)
    return hpq