import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator
def _check_values(self, values):
    if not cp.issubdtype(values.dtype, cp.inexact):
        values = values.astype(float)
    return values