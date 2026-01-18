from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
def _compute_k(self, n):
    if self.k_depth == 'full':
        k = int(np.log2(n)) + 1
    elif self.k_depth == 'tukey':
        k = int(np.log2(n)) - 3
    elif self.k_depth == 'proportion':
        k = int(np.log2(n)) - int(np.log2(n * self.outlier_prop)) + 1
    elif self.k_depth == 'trustworthy':
        normal_quantile_func = np.vectorize(NormalDist().inv_cdf)
        point_conf = 2 * normal_quantile_func(1 - self.trust_alpha / 2) ** 2
        k = int(np.log2(n / point_conf)) + 1
    else:
        k = int(self.k_depth)
    return max(k, 1)