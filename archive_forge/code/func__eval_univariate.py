from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
def _eval_univariate(self, x, weights):
    """Inner function for ECDF of one variable."""
    sorter = x.argsort()
    x = x[sorter]
    weights = weights[sorter]
    y = weights.cumsum()
    if self.stat in ['percent', 'proportion']:
        y = y / y.max()
    if self.stat == 'percent':
        y = y * 100
    x = np.r_[-np.inf, x]
    y = np.r_[0, y]
    if self.complementary:
        y = y.max() - y
    return (y, x)