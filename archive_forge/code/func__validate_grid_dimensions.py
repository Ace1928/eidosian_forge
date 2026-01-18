import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator
def _validate_grid_dimensions(self, points, method):
    k = self._SPLINE_DEGREE_MAP[method]
    for i, point in enumerate(points):
        ndim = len(cp.atleast_1d(point))
        if ndim <= k:
            raise ValueError(f'There are {ndim} points in dimension {i}, but method {method} requires at least  {k + 1} points per dimension.')