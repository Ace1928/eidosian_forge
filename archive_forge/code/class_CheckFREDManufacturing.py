import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.statespace import varmax, sarimax
from statsmodels.iolib.summary import forg
from .results import results_varmax
class CheckFREDManufacturing(CheckVARMAX):

    @classmethod
    def setup_class(cls, true, order, trend, error_cov_type, cov_type='approx', **kwargs):
        cls.true = true
        path = os.path.join(current_path, 'results', 'manufac.dta')
        with open(path, 'rb') as test_data:
            dta = pd.read_stata(test_data)
        dta.index = pd.DatetimeIndex(dta.month, freq='MS')
        dta['dlncaputil'] = dta['lncaputil'].diff()
        dta['dlnhours'] = dta['lnhours'].diff()
        endog = dta.loc['1972-02-01':, ['dlncaputil', 'dlnhours']]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            cls.model = varmax.VARMAX(endog, order=order, trend=trend, error_cov_type=error_cov_type, **kwargs)
        cls.results = cls.model.smooth(true['params'], cov_type=cov_type)