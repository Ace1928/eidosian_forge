from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def _diff_1_if_inf(a, b):
    if a == Infinity or b == Infinity:
        return 1
    return a - b