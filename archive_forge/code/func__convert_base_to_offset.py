from __future__ import annotations
from enum import Enum
from typing import Literal
import pandas as pd
from packaging.version import Version
from xarray.coding import cftime_offsets
def _convert_base_to_offset(base, freq, index):
    """Required until we officially deprecate the base argument to resample.  This
    translates a provided `base` argument to an `offset` argument, following logic
    from pandas.
    """
    from xarray.coding.cftimeindex import CFTimeIndex
    if isinstance(index, pd.DatetimeIndex):
        freq = cftime_offsets._new_to_legacy_freq(freq)
        freq = pd.tseries.frequencies.to_offset(freq)
        if isinstance(freq, pd.offsets.Tick):
            return pd.Timedelta(base * freq.nanos // freq.n)
    elif isinstance(index, CFTimeIndex):
        freq = cftime_offsets.to_offset(freq)
        if isinstance(freq, cftime_offsets.Tick):
            return base * freq.as_timedelta() // freq.n
    else:
        raise ValueError('Can only resample using a DatetimeIndex or CFTimeIndex.')