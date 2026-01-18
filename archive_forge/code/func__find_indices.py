import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator
def _find_indices(self, xi):
    indices = []
    norm_distances = []
    for x, grid in zip(xi, self.grid):
        i = cp.searchsorted(grid, x) - 1
        cp.clip(i, 0, grid.size - 2, i)
        indices.append(i)
        denom = grid[i + 1] - grid[i]
        norm_dist = cp.where(denom != 0, (x - grid[i]) / denom, 0)
        norm_distances.append(norm_dist)
    return (indices, norm_distances)