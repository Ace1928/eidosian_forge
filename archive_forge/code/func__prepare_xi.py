import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator
def _prepare_xi(self, xi):
    ndim = len(self.grid)
    xi = _ndim_coords_from_arrays(xi, ndim=ndim)
    if xi.shape[-1] != len(self.grid):
        raise ValueError(f'The requested sample points xi have dimension {xi.shape[-1]} but this RegularGridInterpolator has dimension {ndim}')
    xi_shape = xi.shape
    xi = xi.reshape(-1, xi_shape[-1])
    xi = cp.asarray(xi, dtype=float)
    is_nans = cp.isnan(xi).T
    nans = is_nans[0].copy()
    for is_nan in is_nans[1:]:
        cp.logical_or(nans, is_nan, nans)
    if self.bounds_error:
        for i, p in enumerate(xi.T):
            if not cp.logical_and(cp.all(self.grid[i][0] <= p), cp.all(p <= self.grid[i][-1])):
                raise ValueError('One of the requested xi is out of bounds in dimension %d' % i)
        out_of_bounds = None
    else:
        out_of_bounds = self._find_out_of_bounds(xi.T)
    return (xi, xi_shape, ndim, nans, out_of_bounds)