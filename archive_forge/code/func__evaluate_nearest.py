import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator
def _evaluate_nearest(self, indices, norm_distances):
    idx_res = [cp.where(yi <= 0.5, i, i + 1) for i, yi in zip(indices, norm_distances)]
    return self.values[tuple(idx_res)]