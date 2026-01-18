from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
class LetterValues:

    def __init__(self, k_depth, outlier_prop, trust_alpha):
        """
        Compute percentiles of a distribution using various tail stopping rules.

        Parameters
        ----------
        k_depth: "tukey", "proportion", "trustworthy", or "full"
            Stopping rule for choosing tail percentiled to show:

            - tukey: Show a similar number of outliers as in a conventional boxplot.
            - proportion: Show approximately `outlier_prop` outliers.
            - trust_alpha: Use `trust_alpha` level for most extreme tail percentile.

        outlier_prop: float
            Parameter for `k_depth="proportion"` setting the expected outlier rate.
        trust_alpha: float
            Parameter for `k_depth="trustworthy"` setting the confidence threshold.

        Notes
        -----
        Based on the proposal in this paper:
        https://vita.had.co.nz/papers/letter-value-plot.pdf

        """
        k_options = ['tukey', 'proportion', 'trustworthy', 'full']
        if isinstance(k_depth, str):
            _check_argument('k_depth', k_options, k_depth)
        elif not isinstance(k_depth, int):
            err = f'The `k_depth` parameter must be either an integer or string (one of {k_options}), not {k_depth!r}.'
            raise TypeError(err)
        self.k_depth = k_depth
        self.outlier_prop = outlier_prop
        self.trust_alpha = trust_alpha

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

    def __call__(self, x):
        """Evaluate the letter values."""
        k = self._compute_k(len(x))
        exp = (np.arange(k + 1, 1, -1), np.arange(2, k + 2))
        levels = k + 1 - np.concatenate([exp[0], exp[1][1:]])
        percentiles = 100 * np.concatenate([0.5 ** exp[0], 1 - 0.5 ** exp[1]])
        if self.k_depth == 'full':
            percentiles[0] = 0
            percentiles[-1] = 100
        values = np.percentile(x, percentiles)
        fliers = np.asarray(x[(x < values.min()) | (x > values.max())])
        median = np.percentile(x, 50)
        return {'k': k, 'levels': levels, 'percs': percentiles, 'values': values, 'fliers': fliers, 'median': median}