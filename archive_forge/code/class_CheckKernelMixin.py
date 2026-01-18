import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pandas as pd
import pytest
from statsmodels.sandbox.nonparametric import kernels
class CheckKernelMixin:
    se_rtol = 0.7
    upp_rtol = 0.1
    low_rtol = 0.2
    low_atol = 0.3

    def test_smoothconf(self):
        kern_name = self.kern_name
        kern = self.kern
        fittedg = np.array([kern.smoothconf(x, y, xi) for xi in xg])
        self.fittedg = fittedg
        res_fitted = results['s_' + kern_name]
        res_se = results['se_' + kern_name]
        crit = 1.9599639845400545
        se = (fittedg[:, 2] - fittedg[:, 1]) / crit
        fitted = fittedg[:, 1]
        assert_allclose(fitted, res_fitted, rtol=5e-07, atol=1e-20)
        assert_allclose(fitted, res_fitted, rtol=0, atol=1e-06)
        self.se = se
        self.res_se = res_se
        se_valid = np.isfinite(res_se)
        assert_allclose(se[se_valid], res_se[se_valid], rtol=self.se_rtol, atol=0.2)
        mask = np.abs(se - res_se) > 0.2 + 0.2 * res_se
        if not hasattr(self, 'se_n_diff'):
            se_n_diff = 40 * 0.125
        else:
            se_n_diff = self.se_n_diff
        assert_array_less(mask.sum(), se_n_diff + 1)
        res_upp = res_fitted + crit * res_se
        res_low = res_fitted - crit * res_se
        self.res_fittedg = np.column_stack((res_low, res_fitted, res_upp))
        assert_allclose(fittedg[se_valid, 2], res_upp[se_valid], rtol=self.upp_rtol, atol=0.2)
        assert_allclose(fittedg[se_valid, 0], res_low[se_valid], rtol=self.low_rtol, atol=self.low_atol)

    @pytest.mark.slow
    @pytest.mark.smoke
    def test_smoothconf_data(self):
        kern = self.kern
        crit = 1.9599639845400545
        fitted_x = np.array([kern.smoothconf(x, y, xi) for xi in x])