from __future__ import annotations
import copy
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import NDFrameT
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
import pandas.core.algorithms as algos
from pandas.core.apply import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.generic import (
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import (
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import (
from pandas.tseries.frequencies import (
from pandas.tseries.offsets import (
def _get_time_delta_bins(self, ax: TimedeltaIndex):
    if not isinstance(ax, TimedeltaIndex):
        raise TypeError(f'axis must be a TimedeltaIndex, but got an instance of {type(ax).__name__}')
    if not isinstance(self.freq, Tick):
        raise ValueError(f"Resampling on a TimedeltaIndex requires fixed-duration `freq`, e.g. '24h' or '3D', not {self.freq}")
    if not len(ax):
        binner = labels = TimedeltaIndex(data=[], freq=self.freq, name=ax.name)
        return (binner, [], labels)
    start, end = (ax.min(), ax.max())
    if self.closed == 'right':
        end += self.freq
    labels = binner = timedelta_range(start=start, end=end, freq=self.freq, name=ax.name)
    end_stamps = labels
    if self.closed == 'left':
        end_stamps += self.freq
    bins = ax.searchsorted(end_stamps, side=self.closed)
    if self.offset:
        labels += self.offset
    return (binner, bins, labels)