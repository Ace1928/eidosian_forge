from __future__ import annotations
from datetime import (
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_inclusive
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import (

        Return sample standard deviation over requested axis.

        Normalized by `N-1` by default. This can be changed using ``ddof``.

        Parameters
        ----------
        axis : int, optional
            Axis for the function to be applied on. For :class:`pandas.Series`
            this parameter is unused and defaults to ``None``.
        ddof : int, default 1
            Degrees of Freedom. The divisor used in calculations is `N - ddof`,
            where `N` represents the number of elements.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is ``NA``, the result
            will be ``NA``.

        Returns
        -------
        Timedelta

        See Also
        --------
        numpy.ndarray.std : Returns the standard deviation of the array elements
            along given axis.
        Series.std : Return sample standard deviation over requested axis.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.date_range('2001-01-01 00:00', periods=3)
        >>> idx
        DatetimeIndex(['2001-01-01', '2001-01-02', '2001-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.std()
        Timedelta('1 days 00:00:00')
        