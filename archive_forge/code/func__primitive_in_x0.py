from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _primitive_in_x0(f):
    """
    Compute the content in `x_0` and the primitive part of a polynomial `f`
    in
    `\\mathbb Q(\\alpha)[x_0, x_1, \\ldots, x_{n-1}] \\cong \\mathbb Q(\\alpha)[x_1, \\ldots, x_{n-1}][x_0]`.
    """
    fring = f.ring
    ring = fring.drop_to_ground(*range(1, fring.ngens))
    dom = ring.domain.ring
    f_ = ring(f.as_expr())
    cont = dom.zero
    for coeff in f_.itercoeffs():
        cont = func_field_modgcd(cont, coeff)[0]
        if cont == dom.one:
            return (cont, f)
    return (cont, f.quo(cont.set_ring(fring)))