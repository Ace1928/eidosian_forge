from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
class TestZeroInflatedModel_probit(CheckGeneric):

    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        exog = sm.add_constant(data.exog[:, 1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:, 0], prepend=False)
        cls.res1 = sm.ZeroInflatedPoisson(data.endog, exog, exog_infl=exog_infl, inflation='probit').fit(method='newton', maxiter=500, disp=False)
        cls.res1._results._attach_nullmodel = True
        cls.init_keys = ['exog_infl', 'exposure', 'inflation', 'offset']
        cls.init_kwds = {'inflation': 'probit'}
        res2 = RandHIE.zero_inflated_poisson_probit
        cls.res2 = res2

    @pytest.mark.skipif(PLATFORM_LINUX32, reason='Fails on 32-bit Linux')
    def test_fit_regularized(self):
        super().test_fit_regularized()