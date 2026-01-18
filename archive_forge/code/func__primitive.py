from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _primitive(f, p):
    """
    Compute the content and the primitive part of a polynomial in
    `\\mathbb{Z}_p[x_0, \\ldots, x_{k-2}, y] \\cong \\mathbb{Z}_p[y][x_0, \\ldots, x_{k-2}]`.

    Parameters
    ==========

    f : PolyElement
        integer polynomial in `\\mathbb{Z}_p[x0, \\ldots, x{k-2}, y]`
    p : Integer
        modulus of `f`

    Returns
    =======

    contf : PolyElement
        integer polynomial in `\\mathbb{Z}_p[y]`, content of `f`
    ppf : PolyElement
        primitive part of `f`, i.e. `\\frac{f}{contf}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _primitive
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)
    >>> p = 3

    >>> f = x**2*y**2 + x**2*y - y**2 - y
    >>> _primitive(f, p)
    (y**2 + y, x**2 - 1)

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x*y*z - y**2*z**2
    >>> _primitive(f, p)
    (z, x*y - y**2*z)

    """
    ring = f.ring
    dom = ring.domain
    k = ring.ngens
    coeffs = {}
    for monom, coeff in f.iterterms():
        if monom[:-1] not in coeffs:
            coeffs[monom[:-1]] = {}
        coeffs[monom[:-1]][monom[-1]] = coeff
    cont = []
    for coeff in iter(coeffs.values()):
        cont = gf_gcd(cont, gf_from_dict(coeff, p, dom), p, dom)
    yring = ring.clone(symbols=ring.symbols[k - 1])
    contf = yring.from_dense(cont).trunc_ground(p)
    return (contf, f.quo(contf.set_ring(ring)))