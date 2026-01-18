from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def _matrix_taking_triple_to_triple(a, b):
    """
    To quote Jeff:

    The formula for the Moebius transformation taking the a[] to the b[]
    is simple enough:

    f(z) = [ (b1*k - b0) * z  +  (b0*a1 - b1*a0*k)] /
           [     (k - 1) * z  +  (a1 - k*a0)      ]

    where

        k = [(b2-b0)/(b2-b1)] * [(a2-a1)/(a2-a0)]
    """
    (a0, a1, a2), (b0, b1, b2) = _normalize_points(a, b)
    ka = (a2 - a1) / (a2 - a0) if a2 != Infinity else 1
    if b1 == Infinity:
        kb, b1kb = (0, -(b2 - b0))
    else:
        kb = (b2 - b0) / (b2 - b1) if b2 != Infinity else 1
        b1kb = b1 * kb
    k = kb * ka
    return matrix([(b1kb * ka - b0, b0 * a1 - a0 * b1kb * ka), (k - 1, a1 - k * a0)])