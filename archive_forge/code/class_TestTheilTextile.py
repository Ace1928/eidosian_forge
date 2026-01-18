import os
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.sandbox.regression.penalized import TheilGLS
class TestTheilTextile:

    @classmethod
    def setup_class(cls):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(cur_dir, 'results', 'theil_textile_predict.csv')
        cls.res_predict = pd.read_csv(filepath, sep=',')
        names = 'year\tlconsump\tlincome\tlprice'.split()
        data = np.array('        1923\t1.99651\t1.98543\t2.00432\n        1924\t1.99564\t1.99167\t2.00043\n        1925\t2\t2\t2\n        1926\t2.04766\t2.02078\t1.95713\n        1927\t2.08707\t2.02078\t1.93702\n        1928\t2.07041\t2.03941\t1.95279\n        1929\t2.08314\t2.04454\t1.95713\n        1930\t2.13354\t2.05038\t1.91803\n        1931\t2.18808\t2.03862\t1.84572\n        1932\t2.18639\t2.02243\t1.81558\n        1933\t2.20003\t2.00732\t1.78746\n        1934\t2.14799\t1.97955\t1.79588\n        1935\t2.13418\t1.98408\t1.80346\n        1936\t2.22531\t1.98945\t1.72099\n        1937\t2.18837\t2.0103\t1.77597\n        1938\t2.17319\t2.00689\t1.77452\n        1939\t2.2188\t2.0162\t1.78746'.split(), float).reshape(-1, 4)
        endog = data[:, 1]
        exog = np.column_stack((data[:, 2:], np.ones(endog.shape[0])))
        r_matrix = np.array([[1, 0, 0], [0, 1, 0]])
        r_mean = [1, -0.7]
        cov_r = np.array([[0.15 ** 2, -0.01], [-0.01, 0.15 ** 2]])
        mod = TheilGLS(endog, exog, r_matrix, q_matrix=r_mean, sigma_prior=cov_r)
        cls.res1 = mod.fit(cov_type='data-prior', use_t=True)
        cls.res1._cache['scale'] = 0.00018334123641580062
        from .results import results_theil_textile as resmodule
        cls.res2 = resmodule.results_theil_textile

    def test_basic(self):
        pt = self.res2.params_table[:, :6].T
        params2, bse2, tvalues2, pvalues2, ci_low, ci_upp = pt
        assert_allclose(self.res1.params, params2, rtol=2e-06)
        corr_fact = 0.9836026210570028
        corr_fact = 0.9737686504146373
        corr_fact = 1
        assert_allclose(self.res1.bse / corr_fact, bse2, rtol=2e-06)
        assert_allclose(self.res1.tvalues * corr_fact, tvalues2, rtol=2e-06)
        ci = self.res1.conf_int()
        assert_allclose(ci[:, 0], ci_low, rtol=0.01)
        assert_allclose(ci[:, 1], ci_upp, rtol=0.01)
        assert_allclose(self.res1.rsquared, self.res2.r2, rtol=2e-06)
        corr_fact = self.res1.df_resid / self.res2.df_r
        assert_allclose(np.sqrt(self.res1.mse_resid * corr_fact), self.res2.rmse, rtol=2e-06)
        assert_allclose(self.res1.fittedvalues, self.res_predict['fittedvalues'], atol=50000000.0)

    def test_other(self):
        tc = self.res1.test_compatibility()
        assert_allclose(np.squeeze(tc[0]), self.res2.compat, rtol=2e-06)
        assert_allclose(np.squeeze(tc[1]), self.res2.pvalue, rtol=2e-06)
        frac = self.res1.share_data()
        assert_allclose(frac, 0.6946116246864239, rtol=2e-06)

    def test_no_penalization(self):
        res_ols = OLS(self.res1.model.endog, self.res1.model.exog).fit()
        res_theil = self.res1.model.fit(pen_weight=0, cov_type='data-prior')
        assert_allclose(res_theil.params, res_ols.params, rtol=1e-10)
        assert_allclose(res_theil.bse, res_ols.bse, rtol=1e-10)

    @pytest.mark.smoke
    def test_summary(self):
        with pytest.warns(UserWarning):
            self.res1.summary()