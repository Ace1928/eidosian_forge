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
def _get_time_bins(self, ax: DatetimeIndex):
    if not isinstance(ax, DatetimeIndex):
        raise TypeError(f'axis must be a DatetimeIndex, but got an instance of {type(ax).__name__}')
    if len(ax) == 0:
        binner = labels = DatetimeIndex(data=[], freq=self.freq, name=ax.name, dtype=ax.dtype)
        return (binner, [], labels)
    first, last = _get_timestamp_range_edges(ax.min(), ax.max(), self.freq, unit=ax.unit, closed=self.closed, origin=self.origin, offset=self.offset)
    binner = labels = date_range(freq=self.freq, start=first, end=last, tz=ax.tz, name=ax.name, ambiguous=True, nonexistent='shift_forward', unit=ax.unit)
    ax_values = ax.asi8
    binner, bin_edges = self._adjust_bin_edges(binner, ax_values)
    bins = lib.generate_bins_dt64(ax_values, bin_edges, self.closed, hasnans=ax.hasnans)
    if self.closed == 'right':
        labels = binner
        if self.label == 'right':
            labels = labels[1:]
    elif self.label == 'right':
        labels = labels[1:]
    if ax.hasnans:
        binner = binner.insert(0, NaT)
        labels = labels.insert(0, NaT)
    if len(bins) < len(labels):
        labels = labels[:len(bins)]
    return (binner, bins, labels)