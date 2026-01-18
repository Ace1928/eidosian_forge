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
def _check_index_type(self, index: pd.Index, allowed: Union[type, tuple[type, ...]]=(pd.DatetimeIndex, pd.PeriodIndex)) -> Union[pd.DatetimeIndex, pd.PeriodIndex]:
    if isinstance(allowed, type):
        allowed = (allowed,)
    if not isinstance(index, allowed):
        if len(allowed) == 1:
            allowed_types = 'a ' + allowed[0].__name__
        else:
            allowed_types = ', '.join((a.__name__ for a in allowed[:-1]))
            if len(allowed) > 2:
                allowed_types += ','
            allowed_types += ' and ' + allowed[-1].__name__
        msg = f'{type(self).__name__} terms can only be computed from {allowed_types}'
        raise TypeError(msg)
    assert isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex))
    return index