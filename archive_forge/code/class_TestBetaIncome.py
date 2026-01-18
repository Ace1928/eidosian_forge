import io
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import patsy
from statsmodels.api import families
from statsmodels.tools.sm_exceptions import (
from statsmodels.othermod.betareg import BetaModel
from .results import results_betareg as resultsb
class TestBetaIncome:

    @classmethod
    def setup_class(cls):
        formula = 'I(food/income) ~ income + persons'
        exog_prec = patsy.dmatrix('~ persons', income)
        mod_income = BetaModel.from_formula(formula, income, exog_precision=exog_prec, link_precision=links.Log())
        res_income = mod_income.fit(method='newton')
        mod_restricted = BetaModel.from_formula(formula, income, link_precision=links.Log())
        res_restricted = mod_restricted.fit(method='newton')
        cls.res1 = res_income
        cls.resr = res_restricted

    def test_score_test(self):
        res1 = self.res1
        resr = self.resr
        params_restr = np.concatenate([resr.params, [0]])
        r_matrix = np.zeros((1, len(params_restr)))
        r_matrix[0, -1] = 1
        exog_prec_extra = res1.model.exog_precision[:, 1:]
        from statsmodels.base._parameter_inference import score_test
        sc1 = score_test(res1, params_constrained=params_restr, k_constraints=1)
        sc2 = score_test(resr, exog_extra=(None, exog_prec_extra))
        assert_allclose(sc2[:2], sc1[:2])
        sc1_hc = score_test(res1, params_constrained=params_restr, k_constraints=1, r_matrix=r_matrix, cov_type='HC0')
        sc2_hc = score_test(resr, exog_extra=(None, exog_prec_extra), cov_type='HC0')
        assert_allclose(sc2_hc[:2], sc1_hc[:2])

    def test_influence(self):
        res1 = self.res1
        from statsmodels.stats.outliers_influence import MLEInfluence
        influ0 = MLEInfluence(res1)
        influ = res1.get_influence()
        attrs = ['cooks_distance', 'd_fittedvalues', 'd_fittedvalues_scaled', 'd_params', 'dfbetas', 'hat_matrix_diag', 'resid_studentized']
        for attr in attrs:
            getattr(influ, attr)
        frame = influ.summary_frame()
        frame0 = influ0.summary_frame()
        assert_allclose(frame, frame0, rtol=1e-13, atol=1e-13)