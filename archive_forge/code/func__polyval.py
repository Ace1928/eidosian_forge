import math
import cupy
from cupy.linalg import lstsq
from cupyx.scipy.ndimage import convolve1d
from ._arraytools import axis_slice
def _polyval(p, x):
    p = cupy.asarray(p)
    x = cupy.asanyarray(x)
    y = cupy.zeros_like(x)
    for pv in p:
        y = y * x + pv
    return y