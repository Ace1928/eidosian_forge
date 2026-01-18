import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
import scipy.stats as stats
from statsmodels.discrete.discrete_model import Logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tools.tools import add_constant
from .results.results_ordinal_model import data_store as ds
class CheckOrdinalModelMixin:

    def test_basic(self):
        n_cat = ds.n_ordinal_cat
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.params[:-n_cat + 1], res2.coefficients_val, atol=0.0002)
        assert_allclose(res1.bse[:-n_cat + 1], res2.coefficients_stdE, rtol=0.003, atol=1e-05)
        assert_allclose(res1.tvalues[:-n_cat + 1], res2.coefficients_tval, rtol=0.003, atol=0.0007)
        assert_allclose(res1.pvalues[:-n_cat + 1], res2.coefficients_pval, rtol=0.009, atol=1e-05)
        assert_allclose(res1.model.transform_threshold_params(res1.params)[1:-1], res2.thresholds, atol=0.0004)
        assert_allclose(res1.predict()[:7, :], res2.prob_pred, atol=5e-05)

    def test_pandas(self):
        res1 = self.res1
        resp = self.resp
        assert_allclose(res1.params, resp.params, atol=1e-10)
        assert_allclose(res1.bse, resp.bse, atol=1e-10)
        assert_allclose(res1.model.endog, resp.model.endog, rtol=1e-10)
        assert_allclose(res1.model.exog, resp.model.exog, rtol=1e-10)

    def test_formula(self):
        res1 = self.res1
        resf = self.resf
        assert_allclose(res1.params, resf.params, atol=5e-05)
        assert_allclose(res1.bse, resf.bse, atol=5e-05)
        assert_allclose(res1.model.endog, resf.model.endog, rtol=1e-10)
        assert_allclose(res1.model.exog, resf.model.exog, rtol=1e-10)

    def test_unordered(self):
        res1 = self.res1
        resf = self.resu
        assert_allclose(res1.params, resf.params, atol=1e-10)
        assert_allclose(res1.bse, resf.bse, atol=1e-10)
        assert_allclose(res1.model.endog, resf.model.endog, rtol=1e-10)
        assert_allclose(res1.model.exog, resf.model.exog, rtol=1e-10)

    def test_results_other(self):
        res1 = self.res1
        resp = self.resp
        param_names_np = ['x1', 'x2', 'x3', '0/1', '1/2']
        param_names_pd = ['pared', 'public', 'gpa', 'unlikely/somewhat likely', 'somewhat likely/very likely']
        assert res1.model.data.param_names == param_names_np
        assert self.resp.model.data.param_names == param_names_pd
        assert self.resp.model.endog_names == 'apply'
        if hasattr(self, 'pred_table'):
            table = res1.pred_table()
            assert_equal(table.values, self.pred_table)
        res1.summary()
        tt = res1.t_test(np.eye(len(res1.params)))
        assert_allclose(tt.pvalue, res1.pvalues, rtol=1e-13)
        tt = resp.t_test(['pared', 'public', 'gpa'])
        assert_allclose(tt.pvalue, res1.pvalues[:3], rtol=1e-13)
        pred = res1.predict(exog=res1.model.exog[-5:])
        fitted = res1.predict()
        assert_allclose(pred, fitted[-5:], rtol=1e-13)
        pred = resp.predict(exog=resp.model.data.orig_exog.iloc[-5:])
        fitted = resp.predict()
        assert_allclose(pred, fitted[-5:], rtol=1e-13)
        dataf = self.resf.model.data.frame
        dataf_df = pd.DataFrame.from_dict(dataf)
        pred = self.resf.predict(exog=dataf_df.iloc[-5:])
        fitted = self.resf.predict()
        assert_allclose(pred, fitted[-5:], rtol=1e-13)
        n, k = res1.model.exog.shape
        assert_equal(self.resf.df_resid, n - (k + 2))
        assert resp.params.index.tolist() == resp.model.exog_names
        assert resp.bse.index.tolist() == resp.model.exog_names