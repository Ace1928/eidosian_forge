import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
class TestDynamicFactor_general_errors(CheckDynamicFactor):
    """
    Test for a dynamic factor model where errors are as general as possible,
    meaning:

    - Errors are vector autocorrelated, VAR(1)
    - Innovations are correlated
    """

    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm_gen.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm_gen_1', 'predict_dfm_gen_2', 'predict_dfm_gen_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm_gen_1', 'dyn_predict_dfm_gen_2', 'dyn_predict_dfm_gen_3']]
        super().setup_class(true, k_factors=1, factor_order=1, error_var=True, error_order=1, error_cov_type='unstructured')

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()
        assert_allclose(bse[:3], self.true['var_oim'][:3], atol=1e-05)
        assert_allclose(bse[-10:], self.true['var_oim'][-10:], atol=0.0003)

    @pytest.mark.skip('Known failure, no sequence of optimizers has been found which can achieve the maximum.')
    def test_mle(self):
        pass

    def test_summary(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']
        assert_equal(len(tables), 2 + self.model.k_endog + self.model.k_factors + self.model.k_endog + 1)
        assert re.search('Model:.*DynamicFactor\\(factors=1, order=1\\)', tables[0])
        assert re.search('.*VAR\\(1\\) errors', tables[0])
        for i in range(self.model.k_endog):
            offset_loading = self.model.k_factors * i
            table = tables[i + 2]
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)
            assert_equal(len(table.split('\n')), 6)
            pattern = 'loading.f1 +' + forg(params[offset_loading + 0], prec=4)
            assert re.search(pattern, table)
        for i in range(self.model.k_factors):
            offset = self.model.k_endog * self.model.k_factors + 6 + i * self.model.k_factors
            table = tables[2 + self.model.k_endog + i]
            name = self.model.endog_names[i]
            assert re.search('Results for factor equation f%d' % (i + 1), table)
            assert_equal(len(table.split('\n')), 6)
            assert re.search('L1.f1 +' + forg(params[offset + 0], prec=4), table)
        for i in range(self.model.k_endog):
            offset = self.model.k_endog * (self.model.k_factors + i) + 6 + self.model.k_factors
            table = tables[2 + self.model.k_endog + self.model.k_factors + i]
            name = self.model.endog_names[i]
            assert re.search('Results for error equation e\\(%s\\)' % name, table)
            assert_equal(len(table.split('\n')), 8)
            for j in range(self.model.k_endog):
                name = self.model.endog_names[j]
                pattern = 'L1.e\\({}\\) +{}'.format(name, forg(params[offset + j], prec=4))
                assert re.search(pattern, table)
        table = tables[2 + self.model.k_endog + self.model.k_factors + self.model.k_endog]
        name = self.model.endog_names[i]
        assert re.search('Error covariance matrix', table)
        assert_equal(len(table.split('\n')), 11)
        offset = self.model.k_endog * self.model.k_factors
        assert re.search('cov.chol\\[1,1\\] +' + forg(params[offset + 0], prec=4), table)
        assert re.search('cov.chol\\[2,1\\] +' + forg(params[offset + 1], prec=4), table)
        assert re.search('cov.chol\\[2,2\\] +' + forg(params[offset + 2], prec=4), table)
        assert re.search('cov.chol\\[3,1\\] +' + forg(params[offset + 3], prec=4), table)
        assert re.search('cov.chol\\[3,2\\] +' + forg(params[offset + 4], prec=4), table)
        assert re.search('cov.chol\\[3,3\\] +' + forg(params[offset + 5], prec=4), table)