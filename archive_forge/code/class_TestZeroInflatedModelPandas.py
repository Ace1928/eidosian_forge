from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
class TestZeroInflatedModelPandas(CheckGeneric):

    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load_pandas()
        cls.endog = data.endog
        cls.data = data
        exog = sm.add_constant(data.exog.iloc[:, 1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog.iloc[:, 0], prepend=False)
        start_params = np.asarray([0.10337834587498942, -1.0459825102508549, -0.08219794475894268, 0.00856917434709146, -0.026795737379474334, 1.4823632430107334])
        model = sm.ZeroInflatedPoisson(data.endog, exog, exog_infl=exog_infl, inflation='logit')
        cls.res1 = model.fit(start_params=start_params, method='newton', maxiter=500, disp=False)
        cls.res1._results._attach_nullmodel = True
        cls.init_keys = ['exog_infl', 'exposure', 'inflation', 'offset']
        cls.init_kwds = {'inflation': 'logit'}
        res2 = RandHIE.zero_inflated_poisson_logit
        cls.res2 = res2

    def test_names(self):
        param_names = ['inflate_lncoins', 'inflate_const', 'idp', 'lpi', 'fmde', 'const']
        assert_array_equal(self.res1.model.exog_names, param_names)
        assert_array_equal(self.res1.params.index.tolist(), param_names)
        assert_array_equal(self.res1.bse.index.tolist(), param_names)
        exog = sm.add_constant(self.data.exog.iloc[:, 1:4], prepend=True)
        exog_infl = sm.add_constant(self.data.exog.iloc[:, 0], prepend=True)
        param_names = ['inflate_const', 'inflate_lncoins', 'const', 'idp', 'lpi', 'fmde']
        model = sm.ZeroInflatedPoisson(self.data.endog, exog, exog_infl=exog_infl, inflation='logit')
        assert_array_equal(model.exog_names, param_names)