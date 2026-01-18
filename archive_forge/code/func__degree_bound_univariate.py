from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _degree_bound_univariate(f, g):
    """
    Compute an upper bound for the degree of the GCD of two univariate
    integer polynomials `f` and `g`.

    The function chooses a suitable prime `p` and computes the GCD of
    `f` and `g` in `\\mathbb{Z}_p[x]`. The choice of `p` guarantees that
    the degree in `\\mathbb{Z}_p[x]` is greater than or equal to the degree
    in `\\mathbb{Z}[x]`.

    Parameters
    ==========

    f : PolyElement
        univariate integer polynomial
    g : PolyElement
        univariate integer polynomial

    """
    gamma = f.ring.domain.gcd(f.LC, g.LC)
    p = 1
    p = nextprime(p)
    while gamma % p == 0:
        p = nextprime(p)
    fp = f.trunc_ground(p)
    gp = g.trunc_ground(p)
    hp = _gf_gcd(fp, gp, p)
    deghp = hp.degree()
    return deghp