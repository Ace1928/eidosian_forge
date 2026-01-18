import os
import numpy.testing as npt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
import statsmodels.sandbox.nonparametric.kernels as kernels
import statsmodels.nonparametric.bandwidths as bandwidths
class TestKdeWeights(CheckKDE):

    @classmethod
    def setup_class(cls):
        res1 = KDE(Xi)
        weights = np.linspace(1, 100, 200)
        res1.fit(kernel='gau', gridsize=50, weights=weights, fft=False, bw='silverman')
        cls.res1 = res1
        fname = os.path.join(curdir, 'results', 'results_kde_weights.csv')
        cls.res_density = np.genfromtxt(open(fname, 'rb'), skip_header=1)

    def test_evaluate(self):
        kde_vals = [self.res1.evaluate(xi) for xi in self.res1.support]
        kde_vals = np.squeeze(kde_vals)
        mask_valid = np.isfinite(kde_vals)
        kde_vals[~mask_valid] = 0
        npt.assert_almost_equal(kde_vals, self.res_density, self.decimal_density)