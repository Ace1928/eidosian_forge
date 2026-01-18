from statsmodels.compat.pandas import (
from abc import ABC, abstractmethod
import datetime as dt
from typing import Optional, Union
from collections.abc import Hashable, Sequence
import numpy as np
import pandas as pd
from scipy.linalg import qr
from statsmodels.iolib.summary import d_or_f
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import freq_to_period
def _weekly_to_loc(self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]) -> np.ndarray:
    if self._freq.freqstr in ('h', 'H'):
        return index.hour + 24 * index.dayofweek
    elif self._freq.freqstr == 'D':
        return index.dayofweek
    else:
        bdays = pd.bdate_range('2000-1-1', periods=10).dayofweek.unique()
        loc = index.dayofweek
        if not loc.isin(bdays).all():
            raise ValueError('freq is B but index contains days that are not business days.')
        return loc