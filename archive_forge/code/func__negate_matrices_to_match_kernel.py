from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def _negate_matrices_to_match_kernel(matrices, G):
    """
    Normalize things so the signs of the matices match SnapPy's default
    This makes the representations stay close as one increases the precision.
    """
    return [_negate_matrix_to_match_kernel(m, matrix(G.SL2C(g))) for m, g in zip(matrices, G.generators())]