from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
def _define_support_bivariate(self, x1, x2, weights):
    """Create a 2D grid of evaluation points."""
    clip = self.clip
    if clip[0] is None or np.isscalar(clip[0]):
        clip = (clip, clip)
    kde = self._fit([x1, x2], weights)
    bw = np.sqrt(np.diag(kde.covariance).squeeze())
    grid1 = self._define_support_grid(x1, bw[0], self.cut, clip[0], self.gridsize)
    grid2 = self._define_support_grid(x2, bw[1], self.cut, clip[1], self.gridsize)
    return (grid1, grid2)