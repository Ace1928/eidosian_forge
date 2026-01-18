from __future__ import annotations
import datetime as dt
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import prefix_mapping
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.datetimes import (
import pandas.core.common as com
from pandas.core.indexes.base import (
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names
from pandas.core.tools.times import to_time
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
def _new_DatetimeIndex(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't
    have arguments and breaks __new__
    """
    if 'data' in d and (not isinstance(d['data'], DatetimeIndex)):
        data = d.pop('data')
        if not isinstance(data, DatetimeArray):
            tz = d.pop('tz')
            freq = d.pop('freq')
            dta = DatetimeArray._simple_new(data, dtype=tz_to_dtype(tz), freq=freq)
        else:
            dta = data
            for key in ['tz', 'freq']:
                if key in d:
                    assert d[key] == getattr(dta, key)
                    d.pop(key)
        result = cls._simple_new(dta, **d)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = cls.__new__(cls, **d)
    return result