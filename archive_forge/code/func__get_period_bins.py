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
def _get_period_bins(self, ax: PeriodIndex):
    if not isinstance(ax, PeriodIndex):
        raise TypeError(f'axis must be a PeriodIndex, but got an instance of {type(ax).__name__}')
    memb = ax.asfreq(self.freq, how=self.convention)
    nat_count = 0
    if memb.hasnans:
        nat_count = np.sum(memb._isnan)
        memb = memb[~memb._isnan]
    if not len(memb):
        bins = np.array([], dtype=np.int64)
        binner = labels = PeriodIndex(data=[], freq=self.freq, name=ax.name)
        if len(ax) > 0:
            binner, bins, labels = _insert_nat_bin(binner, bins, labels, len(ax))
        return (binner, bins, labels)
    freq_mult = self.freq.n
    start = ax.min().asfreq(self.freq, how=self.convention)
    end = ax.max().asfreq(self.freq, how='end')
    bin_shift = 0
    if isinstance(self.freq, Tick):
        p_start, end = _get_period_range_edges(start, end, self.freq, closed=self.closed, origin=self.origin, offset=self.offset)
        start_offset = Period(start, self.freq) - Period(p_start, self.freq)
        bin_shift = start_offset.n % freq_mult
        start = p_start
    labels = binner = period_range(start=start, end=end, freq=self.freq, name=ax.name)
    i8 = memb.asi8
    expected_bins_count = len(binner) * freq_mult
    i8_extend = expected_bins_count - (i8[-1] - i8[0])
    rng = np.arange(i8[0], i8[-1] + i8_extend, freq_mult)
    rng += freq_mult
    rng -= bin_shift
    prng = type(memb._data)(rng, dtype=memb.dtype)
    bins = memb.searchsorted(prng, side='left')
    if nat_count > 0:
        binner, bins, labels = _insert_nat_bin(binner, bins, labels, nat_count)
    return (binner, bins, labels)