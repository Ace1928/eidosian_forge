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
class InfluenceCompareExact:

    def test_basics(self):
        infl1 = self.infl1
        infl0 = self.infl0
        assert_allclose(infl0.hat_matrix_diag, infl1.hat_matrix_diag, rtol=1e-12)
        assert_allclose(infl0.resid_studentized, infl1.resid_studentized, rtol=1e-12, atol=1e-07)
        cd_rtol = getattr(self, 'cd_rtol', 1e-07)
        assert_allclose(infl0.cooks_distance[0], infl1.cooks_distance[0], rtol=cd_rtol, atol=1e-14)
        assert_allclose(infl0.dfbetas, infl1.dfbetas, rtol=1e-09, atol=5e-09)
        assert_allclose(infl0.d_params, infl1.d_params, rtol=1e-09, atol=5e-09)
        assert_allclose(infl0.d_fittedvalues, infl1.d_fittedvalues, rtol=5e-09, atol=1e-14)
        assert_allclose(infl0.d_fittedvalues_scaled, infl1.d_fittedvalues_scaled, rtol=5e-09, atol=1e-14)

    @pytest.mark.smoke
    @pytest.mark.matplotlib
    def test_plots(self, close_figures):
        infl1 = self.infl1
        infl0 = self.infl0
        fig = infl0.plot_influence(external=False)
        fig = infl1.plot_influence(external=False)
        fig = infl0.plot_index('resid', threshold=0.2, title='')
        fig = infl1.plot_index('resid', threshold=0.2, title='')
        fig = infl0.plot_index('dfbeta', idx=1, threshold=0.2, title='')
        fig = infl1.plot_index('dfbeta', idx=1, threshold=0.2, title='')
        fig = infl0.plot_index('cook', idx=1, threshold=0.2, title='')
        fig = infl1.plot_index('cook', idx=1, threshold=0.2, title='')
        fig = infl0.plot_index('hat', idx=1, threshold=0.2, title='')
        fig = infl1.plot_index('hat', idx=1, threshold=0.2, title='')

    def test_summary(self):
        infl1 = self.infl1
        infl0 = self.infl0
        df0 = infl0.summary_frame()
        df1 = infl1.summary_frame()
        assert_allclose(df0.values, df1.values, rtol=5e-05, atol=1e-14)
        pdt.assert_index_equal(df0.index, df1.index)