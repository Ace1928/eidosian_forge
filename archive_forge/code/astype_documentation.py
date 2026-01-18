from __future__ import annotations
import inspect
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import IntCastingNaNError
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
Checks if astype avoided copying the data.

    Parameters
    ----------
    dtype : Original dtype
    new_dtype : target dtype

    Returns
    -------
    True if new data is a view or not guaranteed to be a copy, False otherwise
    