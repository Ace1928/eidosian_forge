from __future__ import annotations
from statsmodels.compat.pandas import (
import numbers
import warnings
import numpy as np
from pandas import (
from pandas.tseries.frequencies import to_offset
from statsmodels.base.data import PandasData
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ValueWarning
class TimeSeriesModelResults(base.LikelihoodModelResults):

    def __init__(self, model, params, normalized_cov_params, scale=1.0):
        self.data = model.data
        super().__init__(model, params, normalized_cov_params, scale)