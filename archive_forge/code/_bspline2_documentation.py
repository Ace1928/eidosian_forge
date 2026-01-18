import operator
from numpy.core.multiarray import normalize_axis_index
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import (
 Construct the interpolating spline spl(x) = y with *full* linalg.

        Only useful for testing, do not call directly!
        This version is O(N**2) in memory and O(N**3) in flop count.
    