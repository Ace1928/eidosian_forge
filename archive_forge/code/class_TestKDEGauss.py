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
class TestKDEGauss(CheckKDE):

    @classmethod
    def setup_class(cls):
        res1 = KDE(Xi)
        res1.fit(kernel='gau', fft=False, bw='silverman')
        cls.res1 = res1
        cls.res_density = KDEResults['gau_d']

    def test_evaluate(self):
        kde_vals = [self.res1.evaluate(xi) for xi in self.res1.support]
        kde_vals = np.squeeze(kde_vals)
        mask_valid = np.isfinite(kde_vals)
        kde_vals[~mask_valid] = 0
        npt.assert_almost_equal(kde_vals, self.res_density, self.decimal_density)

    def test_support_gridded(self):
        kde = self.res1
        support = KCDEResults['gau_support']
        npt.assert_allclose(support, kde.support)

    def test_cdf_gridded(self):
        kde = self.res1
        cdf = KCDEResults['gau_cdf']
        npt.assert_allclose(cdf, kde.cdf)

    def test_sf_gridded(self):
        kde = self.res1
        sf = KCDEResults['gau_sf']
        npt.assert_allclose(sf, kde.sf)

    def test_icdf_gridded(self):
        kde = self.res1
        icdf = KCDEResults['gau_icdf']
        npt.assert_allclose(icdf, kde.icdf)