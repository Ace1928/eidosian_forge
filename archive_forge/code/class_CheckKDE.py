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
class CheckKDE:
    decimal_density = 7

    def test_density(self):
        npt.assert_almost_equal(self.res1.density, self.res_density, self.decimal_density)

    def test_evaluate(self):
        kde_vals = [np.squeeze(self.res1.evaluate(xi)) for xi in self.res1.support]
        kde_vals = np.squeeze(kde_vals)
        mask_valid = np.isfinite(kde_vals)
        kde_vals[~mask_valid] = 0
        npt.assert_almost_equal(kde_vals, self.res_density, self.decimal_density)