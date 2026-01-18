from statsmodels.compat.pandas import testing as pdt
import os.path
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import MLEInfluence
class TestInfluenceGaussianGLMOLS(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        from .test_diagnostic import get_duncan_data
        endog, exog, labels = get_duncan_data()
        data = pd.DataFrame(np.column_stack((endog, exog)), columns='y const var1 var2'.split(), index=labels)
        res0 = GLM.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        res1 = OLS.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        cls.infl1 = res1.get_influence()
        cls.infl0 = res0.get_influence()

    def test_basics(self):
        infl1 = self.infl1
        infl0 = self.infl0
        assert_allclose(infl0.hat_matrix_diag, infl1.hat_matrix_diag, rtol=1e-12)
        assert_allclose(infl0.resid_studentized, infl1.resid_studentized, rtol=1e-12, atol=1e-07)
        assert_allclose(infl0.cooks_distance, infl1.cooks_distance, rtol=1e-07, atol=1e-14)
        assert_allclose(infl0.dfbetas, infl1.dfbetas, rtol=0.1)
        assert_allclose(infl0.d_params, infl1.dfbeta, rtol=1e-09, atol=1e-14)
        assert_allclose(infl0.d_fittedvalues_scaled, infl1.dffits_internal[0], rtol=1e-09)
        assert_allclose(infl0.d_linpred, infl0.d_fittedvalues, rtol=1e-12)
        assert_allclose(infl0.d_linpred_scaled, infl0.d_fittedvalues_scaled, rtol=1e-12)

    def test_summary(self):
        infl1 = self.infl1
        infl0 = self.infl0
        df0 = infl0.summary_frame()
        df1 = infl1.summary_frame()
        cols = ['cooks_d', 'standard_resid', 'hat_diag', 'dffits_internal']
        assert_allclose(df0[cols].values, df1[cols].values, rtol=1e-05)
        pdt.assert_index_equal(df0.index, df1.index)