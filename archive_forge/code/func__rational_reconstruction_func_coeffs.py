from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _rational_reconstruction_func_coeffs(hm, p, m, ring, k):
    """
    Reconstruct every coefficient `c_h` of a polynomial `h` in
    `\\mathbb Z_p(t_k)[t_1, \\ldots, t_{k-1}][x, z]` from the corresponding
    coefficient `c_{h_m}` of a polynomial `h_m` in
    `\\mathbb Z_p[t_1, \\ldots, t_k][x, z] \\cong \\mathbb Z_p[t_k][t_1, \\ldots, t_{k-1}][x, z]`
    such that

    .. math::

        c_{h_m} = c_h \\; \\mathrm{mod} \\, m,

    where `m \\in \\mathbb Z_p[t]`.

    The reconstruction is based on the Euclidean Algorithm. In general, `m`
    is not irreducible, so it is possible that this fails for some
    coefficient. In that case ``None`` is returned.

    Parameters
    ==========

    hm : PolyElement
        polynomial in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`
    p : Integer
        prime number, modulus of `\\mathbb Z_p`
    m : PolyElement
        modulus, polynomial in `\\mathbb Z[t]`, not necessarily irreducible
    ring : PolyRing
        `\\mathbb Z(t_k)[t_1, \\ldots, t_{k-1}][x, z]`, `h` will be an
        element of this ring
    k : Integer
        index of the parameter `t_k` which will be reconstructed

    Returns
    =======

    h : PolyElement
        reconstructed polynomial in
        `\\mathbb Z(t_k)[t_1, \\ldots, t_{k-1}][x, z]` or ``None``

    See also
    ========

    _rational_function_reconstruction

    """
    h = ring.zero
    for monom, coeff in hm.iterterms():
        if k == 0:
            coeffh = _rational_function_reconstruction(coeff, p, m)
            if not coeffh:
                return None
        else:
            coeffh = ring.domain.zero
            for mon, c in coeff.drop_to_ground(k).iterterms():
                ch = _rational_function_reconstruction(c, p, m)
                if not ch:
                    return None
                coeffh[mon] = ch
        h[monom] = coeffh
    return h