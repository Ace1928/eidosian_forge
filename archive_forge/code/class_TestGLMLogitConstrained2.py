from io import StringIO
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import patsy
import pytest
from statsmodels import datasets
from statsmodels.base._constraints import fit_constrained
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
from .results import (
class TestGLMLogitConstrained2(CheckGLMConstrainedMixin):

    @classmethod
    def setup_class(cls):
        cls.idx = slice(None)
        cls.res2 = reslogit.results_constraint2
        mod1 = GLM(spector_data.endog, spector_data.exog, family=families.Binomial())
        constr = 'x1 - x3 = 0'
        cls.res1m = mod1.fit_constrained(constr, atol=1e-10)
        R, q = (cls.res1m.constraints.coefs, cls.res1m.constraints.constants)
        cls.res1 = fit_constrained(mod1, R, q, fit_kwds={'atol': 1e-10})
        cls.constraints_rq = (R, q)

    def test_predict(self):
        res2 = self.res2
        res1 = self.res1m
        predicted = res1.predict()
        assert_allclose(predicted, res2.predict_mu, atol=1e-07)
        assert_allclose(res1.mu, predicted, rtol=1e-10)
        assert_allclose(res1.fittedvalues, predicted, rtol=1e-10)

    @pytest.mark.smoke
    def test_summary(self):
        summ = self.res1m.summary()
        assert_('linear equality constraints' in summ.extra_txt)
        lc_string = str(self.res1m.constraints)
        assert lc_string == 'x1 - x3 = 0.0'

    @pytest.mark.smoke
    def test_summary2(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            summ = self.res1m.summary2()
        assert_('linear equality constraints' in summ.extra_txt[0])

    def test_fit_constrained_wrap(self):
        res2 = self.res2
        from statsmodels.base._constraints import fit_constrained_wrap
        res_wrap = fit_constrained_wrap(self.res1m.model, self.constraints_rq)
        assert_allclose(res_wrap.params, res2.params, rtol=1e-06)
        assert_allclose(res_wrap.params, res2.params, rtol=1e-06)