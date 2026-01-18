from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
class CheckConsistency:
    start_params = None

    def test_cov_type(self):
        mod = self.mod
        res_robust = mod.fit(start_params=self.start_params)
        res_naive = mod.fit(start_params=self.start_params, cov_type='naive')
        res_robust_bc = mod.fit(start_params=self.start_params, cov_type='bias_reduced')
        res_naive.summary()
        res_robust_bc.summary()
        assert_equal(res_robust.cov_type, 'robust')
        assert_equal(res_naive.cov_type, 'naive')
        assert_equal(res_robust_bc.cov_type, 'bias_reduced')
        rtol = 1e-08
        for res, cov_type, cov in [(res_robust, 'robust', res_robust.cov_robust), (res_naive, 'naive', res_robust.cov_naive), (res_robust_bc, 'bias_reduced', res_robust_bc.cov_robust_bc)]:
            bse = np.sqrt(np.diag(cov))
            assert_allclose(res.bse, bse, rtol=rtol)
            if cov_type != 'bias_reduced':
                bse = res_naive.standard_errors(cov_type=cov_type)
                assert_allclose(res.bse, bse, rtol=rtol)
            assert_allclose(res.cov_params(), cov, rtol=rtol, atol=1e-10)
            assert_allclose(res.cov_params_default, cov, rtol=rtol, atol=1e-10)
        assert_(res_robust.cov_params_default is res_robust.cov_robust)
        assert_(res_naive.cov_params_default is res_naive.cov_naive)
        assert_(res_robust_bc.cov_params_default is res_robust_bc.cov_robust_bc)
        assert_raises(ValueError, mod.fit, cov_type='robust_bc')