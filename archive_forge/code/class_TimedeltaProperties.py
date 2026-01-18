from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.accessor import (
from pandas.core.arrays import (
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.base import (
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
@delegate_names(delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_ops, typ='property')
@delegate_names(delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_methods, typ='method')
class TimedeltaProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.

    Examples
    --------
    >>> seconds_series = pd.Series(
    ...     pd.timedelta_range(start="1 second", periods=3, freq="s")
    ... )
    >>> seconds_series
    0   0 days 00:00:01
    1   0 days 00:00:02
    2   0 days 00:00:03
    dtype: timedelta64[ns]
    >>> seconds_series.dt.seconds
    0    1
    1    2
    2    3
    dtype: int32
    """

    def to_pytimedelta(self) -> np.ndarray:
        """
        Return an array of native :class:`datetime.timedelta` objects.

        Python's standard `datetime` library uses a different representation
        timedelta's. This method converts a Series of pandas Timedeltas
        to `datetime.timedelta` format with the same length as the original
        Series.

        Returns
        -------
        numpy.ndarray
            Array of 1D containing data with `datetime.timedelta` type.

        See Also
        --------
        datetime.timedelta : A duration expressing the difference
            between two date, time, or datetime.

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit="d"))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[ns]

        >>> s.dt.to_pytimedelta()
        array([datetime.timedelta(0), datetime.timedelta(days=1),
        datetime.timedelta(days=2), datetime.timedelta(days=3),
        datetime.timedelta(days=4)], dtype=object)
        """
        return self._get_values().to_pytimedelta()

    @property
    def components(self):
        """
        Return a Dataframe of the components of the Timedeltas.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='s'))
        >>> s
        0   0 days 00:00:00
        1   0 days 00:00:01
        2   0 days 00:00:02
        3   0 days 00:00:03
        4   0 days 00:00:04
        dtype: timedelta64[ns]
        >>> s.dt.components
           days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0     0      0        0        0             0             0            0
        1     0      0        0        1             0             0            0
        2     0      0        0        2             0             0            0
        3     0      0        0        3             0             0            0
        4     0      0        0        4             0             0            0
        """
        return self._get_values().components.set_index(self._parent.index).__finalize__(self._parent)

    @property
    def freq(self):
        return self._get_values().inferred_freq