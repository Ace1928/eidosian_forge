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
def _range_from_time_index(self, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    index = self._index
    if isinstance(self._index, pd.PeriodIndex):
        if isinstance(start, pd.Timestamp):
            start = start.to_period(freq=self._index_freq)
        if isinstance(stop, pd.Timestamp):
            stop = stop.to_period(freq=self._index_freq)
    if start < index[0]:
        raise ValueError(START_BEFORE_INDEX_ERR)
    if stop <= self._index[-1]:
        return self.in_sample().loc[start:stop]
    new_idx = self._extend_time_index(stop)
    oos_idx = new_idx[new_idx > index[-1]]
    oos = self.out_of_sample(oos_idx.shape[0], oos_idx)
    if start >= oos_idx[0]:
        return oos.loc[start:stop]
    both = pd.concat([self.in_sample(), oos], axis=0)
    return both.loc[start:stop]