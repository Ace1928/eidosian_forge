from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _to_ANP_poly(f, ring):
    """
    Convert a polynomial
    `f \\in \\mathbb Z[x_1, \\ldots, x_{n-1}][z]/(\\check m_{\\alpha}(z))[x_0]`
    to a polynomial in `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`,
    where `\\check m_{\\alpha}(z) \\in \\mathbb Z[z]` is the primitive associate
    of the minimal polynomial `m_{\\alpha}(z)` of `\\alpha` over
    `\\mathbb Q`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `\\mathbb Z[x_1, \\ldots, x_{n-1}][x_0, z]`
    ring : PolyRing
        `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`

    Returns
    =======

    f_ : PolyElement
        polynomial in `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`

    """
    domain = ring.domain
    f_ = ring.zero
    if isinstance(f.ring.domain, PolynomialRing):
        for monom, coeff in f.iterterms():
            for mon, coef in coeff.iterterms():
                m = (monom[0],) + mon
                c = domain([domain.domain(coef)] + [0] * monom[1])
                if m not in f_:
                    f_[m] = c
                else:
                    f_[m] += c
    else:
        for monom, coeff in f.iterterms():
            m = (monom[0],)
            c = domain([domain.domain(coeff)] + [0] * monom[1])
            if m not in f_:
                f_[m] = c
            else:
                f_[m] += c
    return f_