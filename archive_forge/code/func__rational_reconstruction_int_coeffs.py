from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _rational_reconstruction_int_coeffs(hm, m, ring):
    """
    Reconstruct every rational coefficient `c_h` of a polynomial `h` in
    `\\mathbb Q[t_1, \\ldots, t_k][x, z]` from the corresponding integer
    coefficient `c_{h_m}` of a polynomial `h_m` in
    `\\mathbb Z[t_1, \\ldots, t_k][x, z]` such that

    .. math::

        c_{h_m} = c_h \\; \\mathrm{mod} \\, m,

    where `m \\in \\mathbb Z`.

    The reconstruction is based on the Euclidean Algorithm. In general,
    `m` is not a prime number, so it is possible that this fails for some
    coefficient. In that case ``None`` is returned.

    Parameters
    ==========

    hm : PolyElement
        polynomial in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`
    m : Integer
        modulus, not necessarily prime
    ring : PolyRing
        `\\mathbb Q[t_1, \\ldots, t_k][x, z]`, `h` will be an element of this
        ring

    Returns
    =======

    h : PolyElement
        reconstructed polynomial in `\\mathbb Q[t_1, \\ldots, t_k][x, z]` or
        ``None``

    See also
    ========

    _integer_rational_reconstruction

    """
    h = ring.zero
    if isinstance(ring.domain, PolynomialRing):
        reconstruction = _rational_reconstruction_int_coeffs
        domain = ring.domain.ring
    else:
        reconstruction = _integer_rational_reconstruction
        domain = hm.ring.domain
    for monom, coeff in hm.iterterms():
        coeffh = reconstruction(coeff, m, domain)
        if not coeffh:
            return None
        h[monom] = coeffh
    return h