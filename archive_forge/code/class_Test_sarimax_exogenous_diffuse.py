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
class Test_sarimax_exogenous_diffuse(SARIMAXCoverageTest):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 2, 2)
        kwargs['seasonal_order'] = (3, 2, 2, 4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog)) ** 2
        kwargs['decimal'] = 2
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1000000000.0
        super().setup_class(51, *args, **kwargs)