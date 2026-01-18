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
class TestDynamicFactor2(CheckDynamicFactor):
    """
    Test for a dynamic factor model with two VAR(1) factors
    """

    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm2.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm2_1', 'predict_dfm2_2', 'predict_dfm2_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm2_1', 'dyn_predict_dfm2_2', 'dyn_predict_dfm2_3']]
        super().setup_class(true, k_factors=2, factor_order=1)

    def test_mle(self):
        pass

    def test_bse(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_summary(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']
        assert_equal(len(tables), 2 + self.model.k_endog + self.model.k_factors + 1)
        assert re.search('Model:.*DynamicFactor\\(factors=2, order=1\\)', tables[0])
        for i in range(self.model.k_endog):
            offset_loading = self.model.k_factors * i
            table = tables[i + 2]
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)
            assert_equal(len(table.split('\n')), 7)
            assert re.search('loading.f1 +' + forg(params[offset_loading + 0], prec=4), table)
            assert re.search('loading.f2 +' + forg(params[offset_loading + 1], prec=4), table)
        for i in range(self.model.k_factors):
            offset = self.model.k_endog * (self.model.k_factors + 1) + i * self.model.k_factors
            table = tables[self.model.k_endog + i + 2]
            name = self.model.endog_names[i]
            assert re.search('Results for factor equation f%d' % (i + 1), table)
            assert_equal(len(table.split('\n')), 7)
            assert re.search('L1.f1 +' + forg(params[offset + 0], prec=4), table)
            assert re.search('L1.f2 +' + forg(params[offset + 1], prec=4), table)
        table = tables[2 + self.model.k_endog + self.model.k_factors]
        name = self.model.endog_names[i]
        assert re.search('Error covariance matrix', table)
        assert_equal(len(table.split('\n')), 8)
        offset = self.model.k_endog * self.model.k_factors
        for i in range(self.model.k_endog):
            iname = self.model.endog_names[i]
            iparam = forg(params[offset + i], prec=4)
            assert re.search('sigma2.{} +{}'.format(iname, iparam), table)