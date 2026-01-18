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
class TestVARExtras:

    @classmethod
    def setup_class(cls):
        mdata = macrodata.load_pandas().data
        mdata = mdata[['realgdp', 'realcons', 'realinv']]
        data = mdata.values
        data = np.diff(np.log(data), axis=0) * 400
        cls.res0 = VAR(data).fit(maxlags=2)
        cls.resl1 = VAR(data).fit(maxlags=1)
        cls.data = data

    def test_process(self, close_figures):
        res0 = self.res0
        k_ar = res0.k_ar
        fc20 = res0.forecast(res0.endog[-k_ar:], 20)
        mean_lr = res0.mean()
        assert_allclose(mean_lr, fc20[-1], rtol=0.0005)
        ysim = res0.simulate_var(seed=987128)
        assert_allclose(ysim.mean(0), mean_lr, rtol=0.1)
        assert_allclose(ysim[0], res0.intercept, rtol=1e-10)
        assert_allclose(ysim[1], res0.intercept, rtol=1e-10)
        data = self.data
        resl1 = self.resl1
        y_sim_init = res0.simulate_var(seed=987128, initial_values=data[-k_ar:])
        y_sim_init_2 = res0.simulate_var(seed=987128, initial_values=data[-1])
        assert_allclose(y_sim_init[:k_ar], data[-k_ar:])
        assert_allclose(y_sim_init_2[0], data[-1])
        assert_allclose(y_sim_init_2[k_ar - 1], data[-1])
        y_sim_init_3 = resl1.simulate_var(seed=987128, initial_values=data[-1])
        assert_allclose(y_sim_init_3[0], data[-1])
        n_sim = 900
        ysimz = res0.simulate_var(steps=n_sim, offset=np.zeros((n_sim, 3)), seed=987128)
        zero3 = np.zeros(3)
        assert_allclose(ysimz.mean(0), zero3, atol=0.4)
        assert_allclose(ysimz[0], zero3, atol=1e-10)
        assert_allclose(ysimz[1], zero3, atol=1e-10)
        assert_equal(res0.k_trend, 1)
        assert_equal(res0.k_exog_user, 0)
        assert_equal(res0.k_exog, 1)
        assert_equal(res0.k_ar, 2)
        irf = res0.irf()

    @pytest.mark.matplotlib
    def test_process_plotting(self, close_figures):
        res0 = self.res0
        k_ar = res0.k_ar
        fc20 = res0.forecast(res0.endog[-k_ar:], 20)
        irf = res0.irf()
        res0.plotsim()
        res0.plot_acorr()
        fig = res0.plot_forecast(20)
        fcp = fig.axes[0].get_children()[1].get_ydata()[-20:]
        assert_allclose(fc20[:, 0], fcp, rtol=1e-13)
        fcp = fig.axes[1].get_children()[1].get_ydata()[-20:]
        assert_allclose(fc20[:, 1], fcp, rtol=1e-13)
        fcp = fig.axes[2].get_children()[1].get_ydata()[-20:]
        assert_allclose(fc20[:, 2], fcp, rtol=1e-13)
        fig_asym = irf.plot()
        fig_mc = irf.plot(stderr_type='mc', repl=1000, seed=987128)
        for k in range(3):
            a = fig_asym.axes[1].get_children()[k].get_ydata()
            m = fig_mc.axes[1].get_children()[k].get_ydata()
            assert_allclose(a, m, atol=0.1, rtol=0.9)

    def test_forecast_cov(self):
        res = self.res0
        covfc1 = res.forecast_cov(3)
        assert_allclose(covfc1, res.mse(3), rtol=1e-13)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            covfc2 = res.forecast_cov(3, method='auto')
        assert_allclose(covfc2, covfc1, rtol=0.05)
        res_covfc2 = np.array([[[9.45802013, 4.94142038, 37.1999646], [4.94142038, 7.09273624, 5.66215089], [37.1999646, 5.66215089, 259.61275869]], [[11.30364479, 5.72569141, 49.28744123], [5.72569141, 7.409761, 10.98164091], [49.28744123, 10.98164091, 336.4484723]], [[12.36188803, 6.44426905, 53.54588026], [6.44426905, 7.88850029, 13.96382545], [53.54588026, 13.96382545, 352.19564327]]])
        assert_allclose(covfc2, res_covfc2, atol=1e-06)

    def test_exog(self):
        data = self.res0.model.endog
        res_lin_trend = VAR(data).fit(maxlags=2, trend='ct')
        ex = np.arange(len(data))
        res_lin_trend1 = VAR(data, exog=ex).fit(maxlags=2)
        ex2 = np.arange(len(data))[:, None] ** [0, 1]
        res_lin_trend2 = VAR(data, exog=ex2).fit(maxlags=2, trend='n')
        assert_allclose(res_lin_trend.params, res_lin_trend1.params, rtol=0.005)
        assert_allclose(res_lin_trend.params, res_lin_trend2.params, rtol=0.005)
        assert_allclose(res_lin_trend1.params, res_lin_trend2.params, rtol=1e-10)
        y1 = res_lin_trend.simulate_var(seed=987128)
        y2 = res_lin_trend1.simulate_var(seed=987128)
        y3 = res_lin_trend2.simulate_var(seed=987128)
        assert_allclose(y2.mean(0), y1.mean(0), rtol=1e-12)
        assert_allclose(y3.mean(0), y1.mean(0), rtol=1e-12)
        assert_allclose(y3.mean(0), y2.mean(0), rtol=1e-12)
        h = 10
        fc1 = res_lin_trend.forecast(res_lin_trend.endog[-2:], h)
        exf = np.arange(len(data), len(data) + h)
        fc2 = res_lin_trend1.forecast(res_lin_trend1.endog[-2:], h, exog_future=exf)
        with pytest.raises(ValueError, match='exog_future only has'):
            wrong_exf = np.arange(len(data), len(data) + h // 2)
            res_lin_trend1.forecast(res_lin_trend1.endog[-2:], h, exog_future=wrong_exf)
        exf2 = exf[:, None] ** [0, 1]
        fc3 = res_lin_trend2.forecast(res_lin_trend2.endog[-2:], h, exog_future=exf2)
        assert_allclose(fc2, fc1, rtol=1e-12, atol=1e-12)
        assert_allclose(fc3, fc1, rtol=1e-12, atol=1e-12)
        assert_allclose(fc3, fc2, rtol=1e-12, atol=1e-12)
        fci1 = res_lin_trend.forecast_interval(res_lin_trend.endog[-2:], h)
        exf = np.arange(len(data), len(data) + h)
        fci2 = res_lin_trend1.forecast_interval(res_lin_trend1.endog[-2:], h, exog_future=exf)
        exf2 = exf[:, None] ** [0, 1]
        fci3 = res_lin_trend2.forecast_interval(res_lin_trend2.endog[-2:], h, exog_future=exf2)
        assert_allclose(fci2, fci1, rtol=1e-12, atol=1e-12)
        assert_allclose(fci3, fci1, rtol=1e-12, atol=1e-12)
        assert_allclose(fci3, fci2, rtol=1e-12, atol=1e-12)

    def test_multiple_simulations(self):
        res0 = self.res0
        k_ar = res0.k_ar
        neqs = res0.neqs
        init = self.data[-k_ar:]
        sim1 = res0.simulate_var(seed=987128, steps=10)
        sim2 = res0.simulate_var(seed=987128, steps=10, nsimulations=2)
        assert_equal(sim2.shape, (2, 10, neqs))
        assert_allclose(sim1, sim2[0])
        sim2_init = res0.simulate_var(seed=987128, steps=10, initial_values=init, nsimulations=2)
        assert_allclose(sim2_init[0, :k_ar], init)
        assert_allclose(sim2_init[1, :k_ar], init)