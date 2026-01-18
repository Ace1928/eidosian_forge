from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import MONTH_END, YEAR_END, assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import lrange
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas import DataFrame, Series, date_range
import pytest
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
class TestZivotAndrews(SetupZivotAndrews):

    def test_fail_regression_type(self):
        with pytest.raises(ValueError):
            zivot_andrews(self.fail_mdl, regression='x')

    def test_fail_trim_value(self):
        with pytest.raises(ValueError):
            zivot_andrews(self.fail_mdl, trim=0.5)

    def test_fail_array_shape(self):
        with pytest.raises(ValueError):
            zivot_andrews(np.random.rand(50, 2))

    def test_fail_autolag_type(self):
        with pytest.raises(ValueError):
            zivot_andrews(self.fail_mdl, autolag='None')

    @pytest.mark.parametrize('autolag', ['AIC', 'aic', 'Aic'])
    def test_autolag_case_sensitivity(self, autolag):
        res = zivot_andrews(self.fail_mdl, autolag=autolag)
        assert res[3] == 1

    def test_rgnp_case(self):
        res = zivot_andrews(self.fail_mdl, maxlag=8, regression='c', autolag=None)
        assert_allclose([res[0], res[1], res[4]], [-5.57615, 0.00312, 20], rtol=0.001)

    def test_gnpdef_case(self):
        mdlfile = os.path.join(self.run_dir, 'gnpdef.csv')
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, maxlag=8, regression='c', autolag='t-stat')
        assert_allclose([res[0], res[1], res[3], res[4]], [-4.12155, 0.28024, 5, 40], rtol=0.001)

    def test_stkprc_case(self):
        mdlfile = os.path.join(self.run_dir, 'stkprc.csv')
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, maxlag=8, regression='ct', autolag='t-stat')
        assert_allclose([res[0], res[1], res[3], res[4]], [-5.60689, 0.00894, 1, 65], rtol=0.001)

    def test_rgnpq_case(self):
        mdlfile = os.path.join(self.run_dir, 'rgnpq.csv')
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, maxlag=12, regression='t', autolag='t-stat')
        assert_allclose([res[0], res[1], res[3], res[4]], [-3.02761, 0.63993, 12, 102], rtol=0.001)

    def test_rand10000_case(self):
        mdlfile = os.path.join(self.run_dir, 'rand10000.csv')
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, regression='c', autolag='t-stat')
        assert_allclose([res[0], res[1], res[3], res[4]], [-3.48223, 0.69111, 25, 7071], rtol=0.001)