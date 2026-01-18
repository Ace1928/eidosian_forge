from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
def _define_support_grid(self, x, bw, cut, clip, gridsize):
    """Create the grid of evaluation points depending for vector x."""
    clip_lo = -np.inf if clip[0] is None else clip[0]
    clip_hi = +np.inf if clip[1] is None else clip[1]
    gridmin = max(x.min() - bw * cut, clip_lo)
    gridmax = min(x.max() + bw * cut, clip_hi)
    return np.linspace(gridmin, gridmax, gridsize)