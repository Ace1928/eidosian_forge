from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
def _define_support_univariate(self, x, weights):
    """Create a 1D grid of evaluation points."""
    kde = self._fit(x, weights)
    bw = np.sqrt(kde.covariance.squeeze())
    grid = self._define_support_grid(x, bw, self.cut, self.clip, self.gridsize)
    return grid