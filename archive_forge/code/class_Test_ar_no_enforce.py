import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
class Test_ar_no_enforce(SARIMAXCoverageTest):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        kwargs['enforce_stationarity'] = False
        kwargs['enforce_invertibility'] = False
        kwargs['initial_variance'] = 1000000000.0
        kwargs['loglikelihood_burn'] = 0
        super().setup_class(6, *args, **kwargs)
        cls.model.ssm.loglikelihood_burn = 0

    def test_init_keys_replicate(self):
        mod1 = self.model
        kwargs = self.model._get_init_kwds()
        endog = mod1.data.orig_endog
        exog = mod1.data.orig_exog
        model2 = sarimax.SARIMAX(endog, exog, **kwargs)
        model2.ssm.initialization = mod1.ssm.initialization
        res1 = self.model.filter(self.true_params)
        res2 = model2.filter(self.true_params)
        rtol = 1e-06 if PLATFORM_WIN else 1e-13
        assert_allclose(res2.llf, res1.llf, rtol=rtol)