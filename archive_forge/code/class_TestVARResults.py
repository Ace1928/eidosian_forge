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
class TestVARResults(CheckIRF, CheckFEVD):

    @classmethod
    def setup_class(cls):
        cls.p = 2
        cls.data = get_macrodata()
        cls.model = VAR(cls.data)
        cls.names = cls.model.endog_names
        cls.ref = RResults()
        cls.k = len(cls.ref.names)
        cls.res = cls.model.fit(maxlags=cls.p)
        cls.irf = cls.res.irf(cls.ref.nirfs)
        cls.nahead = cls.ref.nahead
        cls.fevd = cls.res.fevd()

    def test_constructor(self):
        ndarr = self.data.view((float, 3), type=np.ndarray)
        model = VAR(ndarr)
        res = model.fit(self.p)

    def test_names(self):
        assert_equal(self.model.endog_names, self.ref.names)
        model2 = VAR(self.data)
        assert_equal(model2.endog_names, self.ref.names)

    def test_get_eq_index(self):
        assert type(self.res.names) is list
        for i, name in enumerate(self.names):
            idx = self.res.get_eq_index(i)
            idx2 = self.res.get_eq_index(name)
            assert_equal(idx, i)
            assert_equal(idx, idx2)
        with pytest.raises(Exception):
            self.res.get_eq_index('foo')

    @pytest.mark.smoke
    def test_repr(self):
        foo = str(self.res)
        bar = repr(self.res)

    def test_params(self):
        assert_almost_equal(self.res.params, self.ref.params, DECIMAL_3)

    @pytest.mark.smoke
    def test_cov_params(self):
        self.res.cov_params

    @pytest.mark.smoke
    def test_cov_ybar(self):
        self.res.cov_ybar()

    @pytest.mark.smoke
    def test_tstat(self):
        self.res.tvalues

    @pytest.mark.smoke
    def test_pvalues(self):
        self.res.pvalues

    @pytest.mark.smoke
    def test_summary(self):
        summ = self.res.summary()

    def test_detsig(self):
        assert_almost_equal(self.res.detomega, self.ref.detomega)

    def test_aic(self):
        assert_almost_equal(self.res.aic, self.ref.aic)

    def test_bic(self):
        assert_almost_equal(self.res.bic, self.ref.bic)

    def test_hqic(self):
        assert_almost_equal(self.res.hqic, self.ref.hqic)

    def test_fpe(self):
        assert_almost_equal(self.res.fpe, self.ref.fpe)

    def test_lagorder_select(self):
        ics = ['aic', 'fpe', 'hqic', 'bic']
        for ic in ics:
            res = self.model.fit(maxlags=10, ic=ic, verbose=True)
        with pytest.raises(Exception):
            self.model.fit(ic='foo')

    def test_nobs(self):
        assert_equal(self.res.nobs, self.ref.nobs)

    def test_stderr(self):
        assert_almost_equal(self.res.stderr, self.ref.stderr, DECIMAL_4)

    def test_loglike(self):
        assert_almost_equal(self.res.llf, self.ref.loglike)

    def test_ma_rep(self):
        ma_rep = self.res.ma_rep(self.nahead)
        assert_almost_equal(ma_rep, self.ref.ma_rep)

    def test_causality(self):
        causedby = self.ref.causality['causedby']
        for i, name in enumerate(self.names):
            variables = self.names[:i] + self.names[i + 1:]
            result = self.res.test_causality(name, variables, kind='f')
            assert_almost_equal(result.pvalue, causedby[i], DECIMAL_4)
            rng = lrange(self.k)
            rng.remove(i)
            result2 = self.res.test_causality(i, rng, kind='f')
            assert_almost_equal(result.pvalue, result2.pvalue, DECIMAL_12)
            result = self.res.test_causality(name, variables, kind='wald')
        _ = self.res.test_causality(self.names[0], self.names[1])
        _ = self.res.test_causality(0, 1)
        with pytest.raises(Exception):
            self.res.test_causality(0, 1, kind='foo')

    def test_causality_no_lags(self):
        res = VAR(self.data).fit(maxlags=0)
        with pytest.raises(RuntimeError, match='0 lags'):
            res.test_causality(0, 1)

    @pytest.mark.smoke
    def test_select_order(self):
        result = self.model.fit(10, ic='aic', verbose=True)
        result = self.model.fit(10, ic='fpe', verbose=True)
        model = VAR(self.model.endog)
        model.select_order()

    def test_is_stable(self):
        assert self.res.is_stable(verbose=True)

    def test_acf(self):
        acfs = self.res.acf(10)
        acfs = self.res.acf()
        assert len(acfs) == self.p + 1

    def test_acf_2_lags(self):
        c = np.zeros((2, 2, 2))
        c[0] = np.array([[0.2, 0.1], [0.15, 0.15]])
        c[1] = np.array([[0.1, 0.9], [0, 0.1]])
        acf = var_acf(c, np.eye(2), 3)
        gamma = np.zeros((6, 6))
        gamma[:2, :2] = acf[0]
        gamma[2:4, 2:4] = acf[0]
        gamma[4:6, 4:6] = acf[0]
        gamma[2:4, :2] = acf[1].T
        gamma[4:, :2] = acf[2].T
        gamma[:2, 2:4] = acf[1]
        gamma[:2, 4:] = acf[2]
        recovered = np.dot(gamma[:2, 2:], np.linalg.inv(gamma[:4, :4]))
        recovered = [recovered[:, 2 * i:2 * (i + 1)] for i in range(2)]
        recovered = np.array(recovered)
        assert_allclose(recovered, c, atol=1e-07)

    @pytest.mark.smoke
    def test_acorr(self):
        acorrs = self.res.acorr(10)

    @pytest.mark.smoke
    def test_forecast(self):
        self.res.forecast(self.res.endog[-5:], 5)

    @pytest.mark.smoke
    def test_forecast_interval(self):
        y = self.res.endog[:-self.p]
        point, lower, upper = self.res.forecast_interval(y, 5)

    @pytest.mark.matplotlib
    def test_plot_sim(self, close_figures):
        self.res.plotsim(steps=100)

    @pytest.mark.matplotlib
    def test_plot(self, close_figures):
        self.res.plot()

    @pytest.mark.matplotlib
    def test_plot_acorr(self, close_figures):
        self.res.plot_acorr()

    @pytest.mark.matplotlib
    def test_plot_forecast(self, close_figures):
        self.res.plot_forecast(5)

    def test_reorder(self):
        data = self.data.view((float, 3), type=np.ndarray)
        names = self.names
        data2 = np.append(np.append(data[:, 2, None], data[:, 0, None], axis=1), data[:, 1, None], axis=1)
        names2 = []
        names2.append(names[2])
        names2.append(names[0])
        names2.append(names[1])
        res2 = VAR(data2).fit(maxlags=self.p)
        res3 = self.res.reorder(['realinv', 'realgdp', 'realcons'])
        assert_almost_equal(res2.params, res3.params)
        assert_almost_equal(res2.sigma_u, res3.sigma_u)
        assert_almost_equal(res2.bic, res3.bic)
        assert_almost_equal(res2.stderr, res3.stderr)

    def test_pickle(self):
        fh = BytesIO()
        del self.res.model.data.orig_endog
        self.res.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.res.__class__.load(fh)
        assert type(res_unpickled) is type(self.res)