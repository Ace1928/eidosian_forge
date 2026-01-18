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
def _downsample(self, how, **kwargs):
    """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
    if self.kind == 'timestamp':
        return super()._downsample(how, **kwargs)
    orig_how = how
    how = com.get_cython_func(how) or how
    if orig_how != how:
        warn_alias_replacement(self, orig_how, how)
    ax = self.ax
    if is_subperiod(ax.freq, self.freq):
        return self._groupby_and_aggregate(how, **kwargs)
    elif is_superperiod(ax.freq, self.freq):
        if how == 'ohlc':
            return self._groupby_and_aggregate(how)
        return self.asfreq()
    elif ax.freq == self.freq:
        return self.asfreq()
    raise IncompatibleFrequency(f'Frequency {ax.freq} cannot be resampled to {self.freq}, as they are not sub or super periods')