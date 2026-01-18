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
def _boxcox(self):
    if self._use_boxcox is None or self._use_boxcox is False:
        self._lambda = np.nan
        return self._y
    if self._use_boxcox is True:
        y, self._lambda = boxcox(self._y)
    elif isinstance(self._use_boxcox, (int, float)):
        self._lambda = float(self._use_boxcox)
        y = boxcox(self._y, self._use_boxcox)
    else:
        raise TypeError('use_boxcox must be True, False or a float.')
    return y