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
def _initialize_known(self):
    msg = "initialization is 'known' but initial_{0} not given"
    if self._initial_level is None:
        raise ValueError(msg.format('level'))
    excess = 'initial_{0} set but model has no {0} component'
    if self.has_trend and self._initial_trend is None:
        raise ValueError(msg.format('trend'))
    elif not self.has_trend and self._initial_trend is not None:
        raise ValueError(excess.format('trend'))
    if self.has_seasonal and self._initial_seasonal is None:
        raise ValueError(msg.format('seasonal'))
    elif not self.has_seasonal and self._initial_seasonal is not None:
        raise ValueError(excess.format('seasonal'))