from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def _negate_matrix_to_match_kernel(m, snappeaM):
    diff_plus = _matrix_L1_distance_to_kernel(m, snappeaM)
    diff_minus = _matrix_L1_distance_to_kernel(m, -snappeaM)
    if diff_plus < diff_minus:
        return m
    else:
        return -m