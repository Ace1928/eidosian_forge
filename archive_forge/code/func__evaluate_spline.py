import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator
def _evaluate_spline(self, xi, method):
    if xi.ndim == 1:
        xi = xi.reshape((1, xi.size))
    m, n = xi.shape
    axes = tuple(range(self.values.ndim))
    axx = axes[:n][::-1] + axes[n:]
    values = self.values.transpose(axx)
    if method == 'pchip':
        _eval_func = self._do_pchip
    else:
        _eval_func = self._do_spline_fit
    k = self._SPLINE_DEGREE_MAP[method]
    last_dim = n - 1
    first_values = _eval_func(self.grid[last_dim], values, xi[:, last_dim], k)
    shape = (m, *self.values.shape[n:])
    result = cp.empty(shape, dtype=self.values.dtype)
    for j in range(m):
        folded_values = first_values[j, ...]
        for i in range(last_dim - 1, -1, -1):
            folded_values = _eval_func(self.grid[i], folded_values, xi[j, i], k)
        result[j, ...] = folded_values
    return result