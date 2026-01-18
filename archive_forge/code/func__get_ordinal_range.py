from __future__ import annotations
from datetime import timedelta
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.fields import isleapyear_arr
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
import pandas.core.common as com
def _get_ordinal_range(start, end, periods, freq, mult: int=1):
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError('Of the three parameters: start, end, and periods, exactly two must be specified')
    if freq is not None:
        freq = to_offset(freq, is_period=True)
        mult = freq.n
    if start is not None:
        start = Period(start, freq)
    if end is not None:
        end = Period(end, freq)
    is_start_per = isinstance(start, Period)
    is_end_per = isinstance(end, Period)
    if is_start_per and is_end_per and (start.freq != end.freq):
        raise ValueError('start and end must have same freq')
    if start is NaT or end is NaT:
        raise ValueError('start and end must not be NaT')
    if freq is None:
        if is_start_per:
            freq = start.freq
        elif is_end_per:
            freq = end.freq
        else:
            raise ValueError('Could not infer freq from start/end')
        mult = freq.n
    if periods is not None:
        periods = periods * mult
        if start is None:
            data = np.arange(end.ordinal - periods + mult, end.ordinal + 1, mult, dtype=np.int64)
        else:
            data = np.arange(start.ordinal, start.ordinal + periods, mult, dtype=np.int64)
    else:
        data = np.arange(start.ordinal, end.ordinal + 1, mult, dtype=np.int64)
    return (data, freq)