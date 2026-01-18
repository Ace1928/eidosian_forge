from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _degree_bound_bivariate(f, g):
    """
    Compute upper degree bounds for the GCD of two bivariate
    integer polynomials `f` and `g`.

    The GCD is viewed as a polynomial in `\\mathbb{Z}[y][x]` and the
    function returns an upper bound for its degree and one for the degree
    of its content. This is done by choosing a suitable prime `p` and
    computing the GCD of the contents of `f \\; \\mathrm{mod} \\, p` and
    `g \\; \\mathrm{mod} \\, p`. The choice of `p` guarantees that the degree
    of the content in `\\mathbb{Z}_p[y]` is greater than or equal to the
    degree in `\\mathbb{Z}[y]`. To obtain the degree bound in the variable
    `x`, the polynomials are evaluated at `y = a` for a suitable
    `a \\in \\mathbb{Z}_p` and then their GCD in `\\mathbb{Z}_p[x]` is
    computed. If no such `a` exists, i.e. the degree in `\\mathbb{Z}_p[x]`
    is always smaller than the one in `\\mathbb{Z}[y][x]`, then the bound is
    set to the minimum of the degrees of `f` and `g` in `x`.

    Parameters
    ==========

    f : PolyElement
        bivariate integer polynomial
    g : PolyElement
        bivariate integer polynomial

    Returns
    =======

    xbound : Integer
        upper bound for the degree of the GCD of the polynomials `f` and
        `g` in the variable `x`
    ycontbound : Integer
        upper bound for the degree of the content of the GCD of the
        polynomials `f` and `g` in the variable `y`

    References
    ==========

    1. [Monagan00]_

    """
    ring = f.ring
    gamma1 = ring.domain.gcd(f.LC, g.LC)
    gamma2 = ring.domain.gcd(_swap(f, 1).LC, _swap(g, 1).LC)
    badprimes = gamma1 * gamma2
    p = 1
    p = nextprime(p)
    while badprimes % p == 0:
        p = nextprime(p)
    fp = f.trunc_ground(p)
    gp = g.trunc_ground(p)
    contfp, fp = _primitive(fp, p)
    contgp, gp = _primitive(gp, p)
    conthp = _gf_gcd(contfp, contgp, p)
    ycontbound = conthp.degree()
    delta = _gf_gcd(_LC(fp), _LC(gp), p)
    for a in range(p):
        if not delta.evaluate(0, a) % p:
            continue
        fpa = fp.evaluate(1, a).trunc_ground(p)
        gpa = gp.evaluate(1, a).trunc_ground(p)
        hpa = _gf_gcd(fpa, gpa, p)
        xbound = hpa.degree()
        return (xbound, ycontbound)
    return (min(fp.degree(), gp.degree()), ycontbound)