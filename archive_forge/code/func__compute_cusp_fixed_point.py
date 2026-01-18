from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def _compute_cusp_fixed_point(self, cusp):
    """
        Compute fixed point for an incomplete cusp.
        """
    dummy, z, p0, p1 = max(self._compute_cusp_fixed_point_data(cusp), key=lambda d: d[0])
    return (p1 - z * p0) / (1 - z)