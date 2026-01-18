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
def _check_timedeltalike_freq_compat(self, other):
    """
        Arithmetic operations with timedelta-like scalars or array `other`
        are only valid if `other` is an integer multiple of `self.freq`.
        If the operation is valid, find that integer multiple.  Otherwise,
        raise because the operation is invalid.

        Parameters
        ----------
        other : timedelta, np.timedelta64, Tick,
                ndarray[timedelta64], TimedeltaArray, TimedeltaIndex

        Returns
        -------
        multiple : int or ndarray[int64]

        Raises
        ------
        IncompatibleFrequency
        """
    assert self.dtype._is_tick_like()
    dtype = np.dtype(f'm8[{self.dtype._td64_unit}]')
    if isinstance(other, (timedelta, np.timedelta64, Tick)):
        td = np.asarray(Timedelta(other).asm8)
    else:
        td = np.asarray(other)
    try:
        delta = astype_overflowsafe(td, dtype=dtype, copy=False, round_ok=False)
    except ValueError as err:
        raise raise_on_incompatible(self, other) from err
    delta = delta.view('i8')
    return lib.item_from_zerodim(delta)