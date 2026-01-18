from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
class CheckVAR:
    res1 = None
    res2 = None

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_3)

    def test_neqs(self):
        assert_equal(self.res1.neqs, self.res2.neqs)

    def test_nobs(self):
        assert_equal(self.res1.avobs, self.res2.nobs)

    def test_df_eq(self):
        assert_equal(self.res1.df_eq, self.res2.df_eq)

    def test_rmse(self):
        results = self.res1.results
        for i in range(len(results)):
            assert_almost_equal(results[i].mse_resid ** 0.5, eval('self.res2.rmse_' + str(i + 1)), DECIMAL_6)

    def test_rsquared(self):
        results = self.res1.results
        for i in range(len(results)):
            assert_almost_equal(results[i].rsquared, eval('self.res2.rsquared_' + str(i + 1)), DECIMAL_3)

    def test_llf(self):
        results = self.res1.results
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_2)
        for i in range(len(results)):
            assert_almost_equal(results[i].llf, eval('self.res2.llf_' + str(i + 1)), DECIMAL_2)

    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic)

    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic)

    def test_hqic(self):
        assert_almost_equal(self.res1.hqic, self.res2.hqic)

    def test_fpe(self):
        assert_almost_equal(self.res1.fpe, self.res2.fpe)

    def test_detsig(self):
        assert_almost_equal(self.res1.detomega, self.res2.detsig)

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)