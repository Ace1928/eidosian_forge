from __future__ import annotations
from collections import abc
from datetime import (
import sys
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.byteswap import (
from pandas._libs.sas import (
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
from pandas.io.common import get_handle
import pandas.io.sas.sas_constants as const
from pandas.io.sas.sasreader import ReaderBase
def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
    """
    Convert to Timestamp if possible, otherwise to datetime.datetime.
    SAS float64 lacks precision for more than ms resolution so the fit
    to datetime.datetime is ok.

    Parameters
    ----------
    sas_datetimes : {Series, Sequence[float]}
       Dates or datetimes in SAS
    unit : {'d', 's'}
       "d" if the floats represent dates, "s" for datetimes

    Returns
    -------
    Series
       Series of datetime64 dtype or datetime.datetime.
    """
    td = (_sas_origin - _unix_origin).as_unit('s')
    if unit == 's':
        millis = cast_from_unit_vectorized(sas_datetimes._values, unit='s', out_unit='ms')
        dt64ms = millis.view('M8[ms]') + td
        return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
    else:
        vals = np.array(sas_datetimes, dtype='M8[D]') + td
        return pd.Series(vals, dtype='M8[s]', index=sas_datetimes.index, copy=False)