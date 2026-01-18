from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
class TestHoltWinters:

    @classmethod
    def setup_class(cls):
        data = [446.6565229, 454.4733065, 455.662974, 423.6322388, 456.2713279, 440.5880501, 425.3325201, 485.1494479, 506.0481621, 526.7919833, 514.268889, 494.2110193]
        index = ['1996-12-31 00:00:00', '1997-12-31 00:00:00', '1998-12-31 00:00:00', '1999-12-31 00:00:00', '2000-12-31 00:00:00', '2001-12-31 00:00:00', '2002-12-31 00:00:00', '2003-12-31 00:00:00', '2004-12-31 00:00:00', '2005-12-31 00:00:00', '2006-12-31 00:00:00', '2007-12-31 00:00:00']
        oildata_oil = pd.Series(data, index)
        oildata_oil.index = pd.DatetimeIndex(oildata_oil.index, freq=pd.infer_freq(oildata_oil.index))
        cls.oildata_oil = oildata_oil
        data = [17.5534, 21.8601, 23.8866, 26.9293, 26.8885, 28.8314, 30.0751, 30.9535, 30.1857, 31.5797, 32.577569, 33.477398, 39.021581, 41.386432, 41.596552]
        index = ['1990-12-31 00:00:00', '1991-12-31 00:00:00', '1992-12-31 00:00:00', '1993-12-31 00:00:00', '1994-12-31 00:00:00', '1995-12-31 00:00:00', '1996-12-31 00:00:00', '1997-12-31 00:00:00', '1998-12-31 00:00:00', '1999-12-31 00:00:00', '2000-12-31 00:00:00', '2001-12-31 00:00:00', '2002-12-31 00:00:00', '2003-12-31 00:00:00', '2004-12-31 00:00:00']
        air_ausair = pd.Series(data, index)
        air_ausair.index = pd.DatetimeIndex(air_ausair.index, freq=pd.infer_freq(air_ausair.index))
        cls.air_ausair = air_ausair
        data = [263.917747, 268.307222, 260.662556, 266.639419, 277.515778, 283.834045, 290.309028, 292.474198, 300.830694, 309.286657, 318.331081, 329.37239, 338.883998, 339.244126, 328.600632, 314.255385, 314.459695, 321.413779, 329.789292, 346.385165, 352.297882, 348.370515, 417.562922, 417.12357, 417.749459, 412.233904, 411.946817, 394.697075, 401.49927, 408.270468, 414.2428]
        index = ['1970-12-31 00:00:00', '1971-12-31 00:00:00', '1972-12-31 00:00:00', '1973-12-31 00:00:00', '1974-12-31 00:00:00', '1975-12-31 00:00:00', '1976-12-31 00:00:00', '1977-12-31 00:00:00', '1978-12-31 00:00:00', '1979-12-31 00:00:00', '1980-12-31 00:00:00', '1981-12-31 00:00:00', '1982-12-31 00:00:00', '1983-12-31 00:00:00', '1984-12-31 00:00:00', '1985-12-31 00:00:00', '1986-12-31 00:00:00', '1987-12-31 00:00:00', '1988-12-31 00:00:00', '1989-12-31 00:00:00', '1990-12-31 00:00:00', '1991-12-31 00:00:00', '1992-12-31 00:00:00', '1993-12-31 00:00:00', '1994-12-31 00:00:00', '1995-12-31 00:00:00', '1996-12-31 00:00:00', '1997-12-31 00:00:00', '1998-12-31 00:00:00', '1999-12-31 00:00:00', '2000-12-31 00:00:00']
        livestock2_livestock = pd.Series(data, index)
        livestock2_livestock.index = pd.DatetimeIndex(livestock2_livestock.index, freq=pd.infer_freq(livestock2_livestock.index))
        cls.livestock2_livestock = livestock2_livestock
        cls.aust = aust
        cls.start_params = [1.5520372162082909e-09, 2.066338221674873e-18, 1.727109018250519e-09, 50.568333479425036, 0.9129273810171223, 0.83535867, 0.50297119, 0.62439273, 0.67723128]

    def test_predict(self):
        fit1 = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='add', seasonal='mul', initialization_method='estimated').fit(start_params=self.start_params)
        fit2 = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='add', seasonal='mul', initialization_method='estimated').fit(start_params=self.start_params)
        assert_almost_equal(fit1.predict('2011-03-01 00:00:00', '2011-12-01 00:00:00'), [61.3083, 37.373, 46.9652, 51.5578], 3)
        assert_almost_equal(fit2.predict(end='2011-12-01 00:00:00'), [61.3083, 37.373, 46.9652, 51.5578], 3)

    def test_ndarray(self):
        fit1 = ExponentialSmoothing(self.aust.values, seasonal_periods=4, trend='add', seasonal='mul', initialization_method='estimated').fit(start_params=self.start_params)
        assert_almost_equal(fit1.forecast(4), [61.3083, 37.373, 46.9652, 51.5578], 3)

    @pytest.mark.xfail(reason='Optimizer does not converge', strict=False)
    def test_forecast(self):
        fit1 = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='add', seasonal='add').fit(method='bh', use_brute=True)
        assert_almost_equal(fit1.forecast(steps=4), [60.9542, 36.8505, 46.1628, 50.1272], 3)

    def test_simple_exp_smoothing(self):
        fit1 = SimpleExpSmoothing(self.oildata_oil, initialization_method='legacy-heuristic').fit(0.2, optimized=False)
        fit2 = SimpleExpSmoothing(self.oildata_oil, initialization_method='legacy-heuristic').fit(0.6, optimized=False)
        fit3 = SimpleExpSmoothing(self.oildata_oil, initialization_method='estimated').fit()
        assert_almost_equal(fit1.forecast(1), [484.802468], 4)
        assert_almost_equal(fit1.level, [446.6565229, 448.21987962, 449.7084985, 444.49324656, 446.84886283, 445.59670028, 441.54386424, 450.26498098, 461.4216172, 474.49569042, 482.45033014, 484.80246797], 4)
        assert_almost_equal(fit2.forecast(1), [501.837461], 4)
        assert_almost_equal(fit3.forecast(1), [496.493543], 4)
        assert_almost_equal(fit3.params['smoothing_level'], 0.891998, 4)
        assert_almost_equal(fit3.params['initial_level'], 447.47844, 3)

    def test_holt(self):
        fit1 = Holt(self.air_ausair, initialization_method='legacy-heuristic').fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
        fit2 = Holt(self.air_ausair, exponential=True, initialization_method='legacy-heuristic').fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
        fit3 = Holt(self.air_ausair, damped_trend=True, initialization_method='estimated').fit(smoothing_level=0.8, smoothing_trend=0.2)
        assert_almost_equal(fit1.forecast(5), [43.76, 45.59, 47.43, 49.27, 51.1], 2)
        assert_almost_equal(fit1.trend, [3.617628, 3.59006512, 3.33438212, 3.23657639, 2.69263502, 2.46388914, 2.2229097, 1.95959226, 1.47054601, 1.3604894, 1.28045881, 1.20355193, 1.88267152, 2.09564416, 1.83655482], 4)
        assert_almost_equal(fit1.fittedfcast, [21.8601, 22.032368, 25.48461872, 27.54058587, 30.28813356, 30.26106173, 31.58122149, 32.599234, 33.24223906, 32.26755382, 33.07776017, 33.95806605, 34.77708354, 40.05535303, 43.21586036, 43.75696849], 4)
        assert_almost_equal(fit2.forecast(5), [44.6, 47.24, 50.04, 53.01, 56.15], 2)
        assert_almost_equal(fit3.forecast(5), [42.85, 43.81, 44.66, 45.41, 46.06], 2)

    @pytest.mark.smoke
    def test_holt_damp_fit(self):
        fit1 = SimpleExpSmoothing(self.livestock2_livestock, initialization_method='estimated').fit()
        mod4 = Holt(self.livestock2_livestock, damped_trend=True, initialization_method='estimated')
        fit4 = mod4.fit(damping_trend=0.98, method='least_squares')
        mod5 = Holt(self.livestock2_livestock, exponential=True, damped_trend=True, initialization_method='estimated')
        fit5 = mod5.fit()
        assert_almost_equal(fit1.params['smoothing_level'], 1.0, 2)
        assert_almost_equal(fit1.params['smoothing_trend'], np.nan, 2)
        assert_almost_equal(fit1.params['damping_trend'], np.nan, 2)
        assert_almost_equal(fit1.params['initial_level'], 263.96, 1)
        assert_almost_equal(fit1.params['initial_trend'], np.nan, 2)
        assert_almost_equal(fit1.sse, 6761.35, 2)
        assert isinstance(fit1.summary().as_text(), str)
        assert_almost_equal(fit4.params['smoothing_level'], 0.98, 2)
        assert_almost_equal(fit4.params['smoothing_trend'], 0.0, 2)
        assert_almost_equal(fit4.params['damping_trend'], 0.98, 2)
        assert_almost_equal(fit4.params['initial_level'], 257.36, 2)
        assert_almost_equal(fit4.params['initial_trend'], 6.64, 2)
        assert_almost_equal(fit4.sse, 6036.56, 2)
        assert isinstance(fit4.summary().as_text(), str)
        assert_almost_equal(fit5.params['smoothing_level'], 0.97, 2)
        assert_almost_equal(fit5.params['smoothing_trend'], 0.0, 2)
        assert_almost_equal(fit5.params['damping_trend'], 0.98, 2)
        assert_almost_equal(fit5.params['initial_level'], 258.95, 1)
        assert_almost_equal(fit5.params['initial_trend'], 1.04, 2)
        assert_almost_equal(fit5.sse, 6082.0, 0)
        assert isinstance(fit5.summary().as_text(), str)

    def test_holt_damp_r(self):
        mod = Holt(self.livestock2_livestock, damped_trend=True, initialization_method='estimated')
        params = {'smoothing_level': 0.97402626, 'smoothing_trend': 0.00010006, 'damping_trend': 0.98, 'initial_level': 252.59039965, 'initial_trend': 6.90265918}
        with mod.fix_params(params):
            fit = mod.fit(optimized=False)
        for key in params.keys():
            assert_allclose(fit.params[key], params[key])
        with mod.fix_params(params):
            opt_fit = mod.fit(optimized=True)
        assert_allclose(fit.sse, opt_fit.sse)
        assert_allclose(opt_fit.params['initial_trend'], params['initial_trend'])
        alt_params = {k: v for k, v in params.items() if 'level' not in k}
        with mod.fix_params(alt_params):
            alt_fit = mod.fit(optimized=True)
        assert not np.allclose(alt_fit.trend.iloc[0], opt_fit.trend.iloc[0])
        assert_allclose(fit.sse / mod.nobs, 195.4397924865488, atol=0.001)
        desired = [252.5903996514365, 263.7992355246843, 268.3623324350207, 261.0312983437606, 266.6590942700923, 277.3958197247272, 283.8256217863908, 290.2962560621914, 292.5701438129583, 300.7655919939834, 309.2118057241649, 318.2377698496536, 329.223870936255, 338.7709778307978, 339.3669793596703, 329.0127022356033, 314.7684267018998, 314.5948077575944, 321.3612035017972, 329.6924360833211, 346.0712138652086, 352.2534120008911, 348.5862874190927, 415.8839400693967, 417.2018843196238, 417.8435306633725, 412.4857261252961, 412.0647865321129, 395.2500605270393, 401.4367438266322, 408.1907701386275, 414.1814574903921]
        assert_allclose(np.r_[fit.params['initial_level'], fit.level], desired)
        desired = [6.902659175332394, 6.765062519124909, 6.629548973536494, 6.495537532917715, 6.365550989616566, 6.238702070454378, 6.11396047676353, 5.991730467006233, 5.871526257315264, 5.754346516684953, 5.639547926790058, 5.527116419415724, 5.417146212898857, 5.309238662451385, 5.202580636191761, 5.096941655567694, 4.993026494493987, 4.89264548621041, 4.794995106664251, 4.699468310763351, 4.606688340205792, 4.514725879754355, 4.42360016839124, 4.341595902295941, 4.254462303550087, 4.169010676686062, 4.084660399498803, 4.002512751871354, 3.92033229814673, 3.842166514133902, 3.76563019420026, 3.690553892582855]
        assert_allclose(np.r_[fit.params['initial_trend'], fit.trend], desired, atol=0.001)
        desired = [259.3550056432622, 270.4289967934267, 274.8592904290865, 267.39692512602, 272.8973342399166, 283.5097477537724, 289.8173030536191, 296.1681519198575, 298.3242395451272, 306.4048515803347, 314.7385626924191, 323.654343940681, 334.5326742248959, 343.9740317200002, 344.4655083831382, 334.0077050580596, 319.661592666504, 319.3896003340806, 326.0602987063282, 334.2979150278692, 350.5857684386102, 356.6778433630504, 352.9214155841161, 420.1387040536467, 421.3712573771029, 421.9291611265248, 416.4886933168049, 415.9872490289468, 399.0919861792231, 405.2020670104834, 411.8810877289437]
        assert_allclose(fit.fittedvalues, desired, atol=0.001)
        desired = [417.7982003051233, 421.3426082635598, 424.8161280628277, 428.2201774661102, 431.556145881327, 434.8253949282395, 438.0292589942138, 441.1690457788685, 444.2460368278302, 447.2614880558126]
        assert_allclose(fit.forecast(10), desired, atol=0.0001)

    def test_hw_seasonal(self):
        mod = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='additive', seasonal='additive', initialization_method='estimated', use_boxcox=True)
        fit1 = mod.fit()
        assert_almost_equal(fit1.forecast(8), [59.96, 38.63, 47.48, 51.89, 62.81, 41.0, 50.06, 54.57], 2)

    def test_hw_seasonal_add_mul(self):
        mod2 = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='add', seasonal='mul', initialization_method='estimated', use_boxcox=True)
        fit2 = mod2.fit()
        assert_almost_equal(fit2.forecast(8), [61.69, 37.37, 47.22, 52.03, 65.08, 39.34, 49.72, 54.79], 2)
        ExponentialSmoothing(self.aust, seasonal_periods=4, trend='mul', seasonal='add', initialization_method='estimated', use_boxcox=0.0).fit()
        ExponentialSmoothing(self.aust, seasonal_periods=4, trend='multiplicative', seasonal='multiplicative', initialization_method='estimated', use_boxcox=0.0).fit()

    def test_hw_seasonal_buggy(self):
        fit3 = ExponentialSmoothing(self.aust, seasonal_periods=4, seasonal='add', initialization_method='estimated', use_boxcox=True).fit()
        assert_almost_equal(fit3.forecast(8), [59.48719, 35.758854, 44.600641, 47.751384, 59.48719, 35.758854, 44.600641, 47.751384], 2)
        fit4 = ExponentialSmoothing(self.aust, seasonal_periods=4, seasonal='mul', initialization_method='estimated', use_boxcox=True).fit()
        assert_almost_equal(fit4.forecast(8), [59.26155037, 35.27811302, 44.00438543, 47.97732693, 59.26155037, 35.27811302, 44.00438543, 47.97732693], 2)