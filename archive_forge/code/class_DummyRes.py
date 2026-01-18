from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import statsmodels.datasets
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.statespace.exponential_smoothing import (
class DummyRes:

    def __init__(self, res):
        self._res = res

    def forecast(self, *args, **kwargs):
        return self._res.forecast(*args, **kwargs)

    def get_prediction(self, *args, **kwargs):
        pred = self._res.get_prediction(*args, **kwargs)
        return DummyPred(pred.predicted_mean, pred.row_labels)