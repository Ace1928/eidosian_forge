from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def _compute_midpoint_helper(b, c, offset):
    height = abs(c - b) * offset
    return FinitePoint(b, height)