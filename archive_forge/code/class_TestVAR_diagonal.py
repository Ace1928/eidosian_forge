import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.statespace import varmax, sarimax
from statsmodels.iolib.summary import forg
from .results import results_varmax
class TestVAR_diagonal(CheckLutkepohl):

    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_diag.copy()
        true['predict'] = var_results.iloc[1:][['predict_diag1', 'predict_diag2', 'predict_diag3']]
        true['dynamic_predict'] = var_results.iloc[1:][['dyn_predict_diag1', 'dyn_predict_diag2', 'dyn_predict_diag3']]
        super().setup_class(true, order=(1, 0), trend='n', error_cov_type='diagonal')

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal() ** 0.5
        assert_allclose(bse ** 2, self.true['var_oim'], atol=1e-05)

    def test_bse_oim(self):
        bse = self.results._cov_params_oim().diagonal() ** 0.5
        assert_allclose(bse ** 2, self.true['var_oim'], atol=0.01)

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']
        assert re.search('Model:.*VAR\\(1\\)', tables[0])
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog
            table = tables[i + 2]
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)
            assert len(table.split('\n')) == 8
            assert re.search('L1.dln_inv +%.4f' % params[offset + 0], table)
            assert re.search('L1.dln_inc +%.4f' % params[offset + 1], table)
            assert re.search('L1.dln_consump +%.4f' % params[offset + 2], table)
        table = tables[-1]
        assert re.search('Error covariance matrix', table)
        assert len(table.split('\n')) == 8
        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert re.search('{} +{:.4f}'.format(names[i], params[i]), table)