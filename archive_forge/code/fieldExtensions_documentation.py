from .polynomial import Polynomial, Monomial
from . import matrix

    This is returning the same as pari's
    rnfequation(base_poly, extension_poly, flag = 3) but
    assumes that base_poly and extension_poly are irreducible
    and thus is much faster.
    