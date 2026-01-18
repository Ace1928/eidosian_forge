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
class CheckKDEWeights:

    @classmethod
    def setup_class(cls):
        cls.x = x = KDEWResults['x']
        weights = KDEWResults['weights']
        res1 = KDE(x)
        res1.fit(kernel=cls.kernel_name, weights=weights, fft=False, bw='scott')
        cls.res1 = res1
        cls.res_density = KDEWResults[cls.res_kernel_name]
    decimal_density = 7

    @pytest.mark.xfail(reason='Not almost equal to 7 decimals', raises=AssertionError, strict=True)
    def test_density(self):
        npt.assert_almost_equal(self.res1.density, self.res_density, self.decimal_density)

    def test_evaluate(self):
        if self.kernel_name == 'cos':
            pytest.skip('Cosine kernel fails against Stata')
        kde_vals = [self.res1.evaluate(xi) for xi in self.x]
        kde_vals = np.squeeze(kde_vals)
        npt.assert_almost_equal(kde_vals, self.res_density, self.decimal_density)

    def test_compare(self):
        xx = self.res1.support
        kde_vals = [np.squeeze(self.res1.evaluate(xi)) for xi in xx]
        kde_vals = np.squeeze(kde_vals)
        mask_valid = np.isfinite(kde_vals)
        kde_vals[~mask_valid] = 0
        npt.assert_almost_equal(self.res1.density, kde_vals, self.decimal_density)
        nobs = len(self.res1.endog)
        kern = self.res1.kernel
        v = kern.density_var(kde_vals, nobs)
        v_direct = kde_vals * kern.L2Norm / kern.h / nobs
        npt.assert_allclose(v, v_direct, rtol=1e-10)
        ci = kern.density_confint(kde_vals, nobs)
        crit = 1.9599639845400545
        hw = kde_vals - ci[:, 0]
        npt.assert_allclose(hw, crit * np.sqrt(v), rtol=1e-10)
        hw = ci[:, 1] - kde_vals
        npt.assert_allclose(hw, crit * np.sqrt(v), rtol=1e-10)

    def test_kernel_constants(self):
        kern = self.res1.kernel
        nc = kern.norm_const
        kern._norm_const = None
        nc2 = kern.norm_const
        npt.assert_allclose(nc, nc2, rtol=1e-10)
        l2n = kern.L2Norm
        kern._L2Norm = None
        l2n2 = kern.L2Norm
        npt.assert_allclose(l2n, l2n2, rtol=1e-10)
        v = kern.kernel_var
        kern._kernel_var = None
        v2 = kern.kernel_var
        npt.assert_allclose(v, v2, rtol=1e-10)