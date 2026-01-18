import numpy as np
from numpy.testing import (
import pytest
from scipy import stats
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.sm_exceptions import InvalidTestWarning
from statsmodels.tools.tools import add_constant
from .results import (
class TestWLSOLSRobustSmall:

    @classmethod
    def setup_class(cls):
        from statsmodels.datasets import grunfeld
        dtapa = grunfeld.data.load_pandas()
        dtapa_endog = dtapa.endog[:200]
        dtapa_exog = dtapa.exog[:200]
        exog = add_constant(dtapa_exog[['value', 'capital']], prepend=False)
        cls.res_wls = WLS(dtapa_endog, exog, weights=1 / dtapa_exog['value']).fit()
        w_sqrt = 1 / np.sqrt(np.asarray(dtapa_exog['value']))
        cls.res_ols = OLS(dtapa_endog * w_sqrt, np.asarray(exog) * w_sqrt[:, None]).fit()
        ids = np.asarray(dtapa_exog[['firm']], 'S20')
        firm_names, firm_id = np.unique(ids, return_inverse=True)
        cls.groups = firm_id
        time = np.require(dtapa_exog[['year']], requirements='W')
        time -= time.min()
        cls.time = np.squeeze(time).astype(int)
        cls.tidx = [(i * 20, 20 * (i + 1)) for i in range(10)]

    def test_all(self):
        all_cov = [('HC0', dict(use_t=True)), ('HC1', dict(use_t=True)), ('HC2', dict(use_t=True)), ('HC3', dict(use_t=True))]
        for cov_type, kwds in all_cov:
            res1 = self.res_ols.get_robustcov_results(cov_type, **kwds)
            res2 = self.res_wls.get_robustcov_results(cov_type, **kwds)
            assert_allclose(res1.params, res2.params, rtol=1e-13)
            assert_allclose(res1.cov_params(), res2.cov_params(), rtol=1e-13)
            assert_allclose(res1.bse, res2.bse, rtol=1e-13)
            assert_allclose(res1.pvalues, res2.pvalues, rtol=1e-13)
            mat = np.eye(len(res1.params))
            ft1 = res1.f_test(mat)
            ft2 = res2.f_test(mat)
            assert_allclose(ft1.fvalue, ft2.fvalue, rtol=1e-12)
            assert_allclose(ft1.pvalue, ft2.pvalue, rtol=5e-11)

    def test_fixed_scale(self):
        cov_type = 'fixed_scale'
        kwds = {}
        res1 = self.res_ols.get_robustcov_results(cov_type, **kwds)
        res2 = self.res_wls.get_robustcov_results(cov_type, **kwds)
        assert_allclose(res1.params, res2.params, rtol=1e-13)
        assert_allclose(res1.cov_params(), res2.cov_params(), rtol=1e-13)
        assert_allclose(res1.bse, res2.bse, rtol=1e-13)
        assert_allclose(res1.pvalues, res2.pvalues, rtol=1e-12)
        tt = res2.t_test(np.eye(len(res2.params)), cov_p=res2.normalized_cov_params)
        assert_allclose(res2.cov_params(), res2.normalized_cov_params, rtol=1e-13)
        assert_allclose(res2.bse, tt.sd, rtol=1e-13)
        assert_allclose(res2.pvalues, tt.pvalue, rtol=1e-13)
        assert_allclose(res2.tvalues, tt.tvalue, rtol=1e-13)
        mod = self.res_wls.model
        mod3 = WLS(mod.endog, mod.exog, weights=mod.weights)
        res3 = mod3.fit(cov_type=cov_type, cov_kwds=kwds)
        tt = res3.t_test(np.eye(len(res3.params)), cov_p=res3.normalized_cov_params)
        assert_allclose(res3.cov_params(), res3.normalized_cov_params, rtol=1e-13)
        assert_allclose(res3.bse, tt.sd, rtol=1e-13)
        assert_allclose(res3.pvalues, tt.pvalue, rtol=1e-13)
        assert_allclose(res3.tvalues, tt.tvalue, rtol=1e-13)