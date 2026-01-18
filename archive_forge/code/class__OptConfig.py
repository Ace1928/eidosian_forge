from statsmodels.compat.pandas import deprecate_kwarg
import contextlib
from typing import Any
from collections.abc import Hashable, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, least_squares, minimize
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.tools.validation import (
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import (
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import HoltWintersArgs
from statsmodels.tsa.holtwinters._smoothers import (
from statsmodels.tsa.holtwinters.results import (
from statsmodels.tsa.tsatools import freq_to_period
class _OptConfig:
    alpha: float
    beta: float
    phi: float
    gamma: float
    level: float
    trend: float
    seasonal: np.ndarray
    y: np.ndarray
    params: np.ndarray
    mask: np.ndarray
    mle_retvals: Any

    def unpack_parameters(self, params) -> '_OptConfig':
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        self.level = params[3]
        self.trend = params[4]
        self.phi = params[5]
        self.seasonal = params[6:]
        return self