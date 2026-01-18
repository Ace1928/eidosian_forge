from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
def _cart2polar_2pi(x, y):
    """convert cartesian coordinates to polar (uses non-standard theta range!)

    NON-STANDARD RANGE! Maps to ``(0, 2*pi)`` rather than usual ``(-pi, +pi)``
    """
    r, t = (np.hypot(x, y), np.arctan2(y, x))
    t += np.where(t < 0.0, 2 * np.pi, 0)
    return (r, t)