from __future__ import annotations
import datetime as dt
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import prefix_mapping
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.datetimes import (
import pandas.core.common as com
from pandas.core.indexes.base import (
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names
from pandas.core.tools.times import to_time
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
@inherit_names(DatetimeArray._field_ops + [method for method in DatetimeArray._datetimelike_methods if method not in ('tz_localize', 'tz_convert', 'strftime')], DatetimeArray, wrap=True)
@inherit_names(['is_normalized'], DatetimeArray, cache=True)
@inherit_names(['tz', 'tzinfo', 'dtype', 'to_pydatetime', 'date', 'time', 'timetz', 'std'] + DatetimeArray._bool_ops, DatetimeArray)
class DatetimeIndex(DatetimeTimedeltaMixin):
    """
    Immutable ndarray-like of datetime64 data.

    Represented internally as int64, and which can be boxed to Timestamp objects
    that are subclasses of datetime and carry metadata.

    .. versionchanged:: 2.0.0
        The various numeric date/time attributes (:attr:`~DatetimeIndex.day`,
        :attr:`~DatetimeIndex.month`, :attr:`~DatetimeIndex.year` etc.) now have dtype
        ``int32``. Previously they had dtype ``int64``.

    Parameters
    ----------
    data : array-like (1-dimensional)
        Datetime-like data to construct index with.
    freq : str or pandas offset object, optional
        One of pandas date offset strings or corresponding objects. The string
        'infer' can be passed in order to set the frequency of the index as the
        inferred frequency upon creation.
    tz : pytz.timezone or dateutil.tz.tzfile or datetime.tzinfo or str
        Set the Timezone of the data.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

        .. deprecated:: 2.1.0

    closed : {'left', 'right'}, optional
        Set whether to include `start` and `end` that are on the
        boundary. The default includes boundary points on either end.

        .. deprecated:: 2.1.0

    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
        When clocks moved backward due to DST, ambiguous times may arise.
        For example in Central European Time (UTC+01), when going from 03:00
        DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
        and at 01:30:00 UTC. In such a situation, the `ambiguous` parameter
        dictates how ambiguous times should be handled.

        - 'infer' will attempt to infer fall dst-transition hours based on
          order
        - bool-ndarray where True signifies a DST time, False signifies a
          non-DST time (note that this flag is only applicable for ambiguous
          times)
        - 'NaT' will return NaT where there are ambiguous times
        - 'raise' will raise an AmbiguousTimeError if there are ambiguous times.
    dayfirst : bool, default False
        If True, parse dates in `data` with the day first order.
    yearfirst : bool, default False
        If True parse dates in `data` with the year first order.
    dtype : numpy.dtype or DatetimeTZDtype or str, default None
        Note that the only NumPy dtype allowed is `datetime64[ns]`.
    copy : bool, default False
        Make a copy of input ndarray.
    name : label, default None
        Name to be stored in the index.

    Attributes
    ----------
    year
    month
    day
    hour
    minute
    second
    microsecond
    nanosecond
    date
    time
    timetz
    dayofyear
    day_of_year
    dayofweek
    day_of_week
    weekday
    quarter
    tz
    freq
    freqstr
    is_month_start
    is_month_end
    is_quarter_start
    is_quarter_end
    is_year_start
    is_year_end
    is_leap_year
    inferred_freq

    Methods
    -------
    normalize
    strftime
    snap
    tz_convert
    tz_localize
    round
    floor
    ceil
    to_period
    to_pydatetime
    to_series
    to_frame
    month_name
    day_name
    mean
    std

    See Also
    --------
    Index : The base pandas Index type.
    TimedeltaIndex : Index of timedelta64 data.
    PeriodIndex : Index of Period data.
    to_datetime : Convert argument to datetime.
    date_range : Create a fixed-frequency DatetimeIndex.

    Notes
    -----
    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
    >>> idx
    DatetimeIndex(['2020-01-01 10:00:00+00:00', '2020-02-01 11:00:00+00:00'],
    dtype='datetime64[ns, UTC]', freq=None)
    """
    _typ = 'datetimeindex'
    _data_cls = DatetimeArray
    _supports_partial_string_indexing = True

    @property
    def _engine_type(self) -> type[libindex.DatetimeEngine]:
        return libindex.DatetimeEngine
    _data: DatetimeArray
    _values: DatetimeArray
    tz: dt.tzinfo | None

    @doc(DatetimeArray.strftime)
    def strftime(self, date_format) -> Index:
        arr = self._data.strftime(date_format)
        return Index(arr, name=self.name, dtype=object)

    @doc(DatetimeArray.tz_convert)
    def tz_convert(self, tz) -> Self:
        arr = self._data.tz_convert(tz)
        return type(self)._simple_new(arr, name=self.name, refs=self._references)

    @doc(DatetimeArray.tz_localize)
    def tz_localize(self, tz, ambiguous: TimeAmbiguous='raise', nonexistent: TimeNonexistent='raise') -> Self:
        arr = self._data.tz_localize(tz, ambiguous, nonexistent)
        return type(self)._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_period)
    def to_period(self, freq=None) -> PeriodIndex:
        from pandas.core.indexes.api import PeriodIndex
        arr = self._data.to_period(freq)
        return PeriodIndex._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_julian_date)
    def to_julian_date(self) -> Index:
        arr = self._data.to_julian_date()
        return Index._simple_new(arr, name=self.name)

    @doc(DatetimeArray.isocalendar)
    def isocalendar(self) -> DataFrame:
        df = self._data.isocalendar()
        return df.set_index(self)

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        return self._data._resolution_obj

    def __new__(cls, data=None, freq: Frequency | lib.NoDefault=lib.no_default, tz=lib.no_default, normalize: bool | lib.NoDefault=lib.no_default, closed=lib.no_default, ambiguous: TimeAmbiguous='raise', dayfirst: bool=False, yearfirst: bool=False, dtype: Dtype | None=None, copy: bool=False, name: Hashable | None=None) -> Self:
        if closed is not lib.no_default:
            warnings.warn(f"The 'closed' keyword in {cls.__name__} construction is deprecated and will be removed in a future version.", FutureWarning, stacklevel=find_stack_level())
        if normalize is not lib.no_default:
            warnings.warn(f"The 'normalize' keyword in {cls.__name__} construction is deprecated and will be removed in a future version.", FutureWarning, stacklevel=find_stack_level())
        if is_scalar(data):
            cls._raise_scalar_data_error(data)
        name = maybe_extract_name(name, data, cls)
        if isinstance(data, DatetimeArray) and freq is lib.no_default and (tz is lib.no_default) and (dtype is None):
            if copy:
                data = data.copy()
            return cls._simple_new(data, name=name)
        dtarr = DatetimeArray._from_sequence_not_strict(data, dtype=dtype, copy=copy, tz=tz, freq=freq, dayfirst=dayfirst, yearfirst=yearfirst, ambiguous=ambiguous)
        refs = None
        if not copy and isinstance(data, (Index, ABCSeries)):
            refs = data._references
        subarr = cls._simple_new(dtarr, name=name, refs=refs)
        return subarr

    @cache_readonly
    def _is_dates_only(self) -> bool:
        """
        Return a boolean if we are only dates (and don't have a timezone)

        Returns
        -------
        bool
        """
        if isinstance(self.freq, Tick):
            delta = Timedelta(self.freq)
            if delta % dt.timedelta(days=1) != dt.timedelta(days=0):
                return False
        return self._values._is_dates_only

    def __reduce__(self):
        d = {'data': self._data, 'name': self.name}
        return (_new_DatetimeIndex, (type(self), d), None)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        if self.tz is not None:
            return isinstance(dtype, DatetimeTZDtype)
        return lib.is_np_dtype(dtype, 'M')

    @cache_readonly
    def _formatter_func(self):
        from pandas.io.formats.format import get_format_datetime64
        formatter = get_format_datetime64(is_dates_only=self._is_dates_only)
        return lambda x: f"'{formatter(x)}'"

    def _can_range_setop(self, other) -> bool:
        if self.tz is not None and (not timezones.is_utc(self.tz)) and (not timezones.is_fixed_offset(self.tz)):
            return False
        if other.tz is not None and (not timezones.is_utc(other.tz)) and (not timezones.is_fixed_offset(other.tz)):
            return False
        return super()._can_range_setop(other)

    def _get_time_micros(self) -> npt.NDArray[np.int64]:
        """
        Return the number of microseconds since midnight.

        Returns
        -------
        ndarray[int64_t]
        """
        values = self._data._local_timestamps()
        ppd = periods_per_day(self._data._creso)
        frac = values % ppd
        if self.unit == 'ns':
            micros = frac // 1000
        elif self.unit == 'us':
            micros = frac
        elif self.unit == 'ms':
            micros = frac * 1000
        elif self.unit == 's':
            micros = frac * 1000000
        else:
            raise NotImplementedError(self.unit)
        micros[self._isnan] = -1
        return micros

    def snap(self, freq: Frequency='S') -> DatetimeIndex:
        """
        Snap time stamps to nearest occurring frequency.

        Returns
        -------
        DatetimeIndex

        Examples
        --------
        >>> idx = pd.DatetimeIndex(['2023-01-01', '2023-01-02',
        ...                        '2023-02-01', '2023-02-02'])
        >>> idx
        DatetimeIndex(['2023-01-01', '2023-01-02', '2023-02-01', '2023-02-02'],
        dtype='datetime64[ns]', freq=None)
        >>> idx.snap('MS')
        DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01', '2023-02-01'],
        dtype='datetime64[ns]', freq=None)
        """
        freq = to_offset(freq)
        dta = self._data.copy()
        for i, v in enumerate(self):
            s = v
            if not freq.is_on_offset(s):
                t0 = freq.rollback(s)
                t1 = freq.rollforward(s)
                if abs(s - t0) < abs(t1 - s):
                    s = t0
                else:
                    s = t1
            dta[i] = s
        return DatetimeIndex._simple_new(dta, name=self.name)

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: dt.datetime):
        """
        Calculate datetime bounds for parsed time string and its resolution.

        Parameters
        ----------
        reso : Resolution
            Resolution provided by parsed string.
        parsed : datetime
            Datetime from parsed string.

        Returns
        -------
        lower, upper: pd.Timestamp
        """
        freq = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
        per = Period(parsed, freq=freq)
        start, end = (per.start_time, per.end_time)
        start = start.tz_localize(parsed.tzinfo)
        end = end.tz_localize(parsed.tzinfo)
        if parsed.tzinfo is not None:
            if self.tz is None:
                raise ValueError('The index must be timezone aware when indexing with a date string with a UTC offset')
        return (start, end)

    def _parse_with_reso(self, label: str):
        parsed, reso = super()._parse_with_reso(label)
        parsed = Timestamp(parsed)
        if self.tz is not None and parsed.tzinfo is None:
            parsed = parsed.tz_localize(self.tz)
        return (parsed, reso)

    def _disallow_mismatched_indexing(self, key) -> None:
        """
        Check for mismatched-tzawareness indexing and re-raise as KeyError.
        """
        try:
            self._data._assert_tzawareness_compat(key)
        except TypeError as err:
            raise KeyError(key) from err

    def get_loc(self, key):
        """
        Get integer location for requested label

        Returns
        -------
        loc : int
        """
        self._check_indexing_error(key)
        orig_key = key
        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT
        if isinstance(key, self._data._recognized_scalars):
            self._disallow_mismatched_indexing(key)
            key = Timestamp(key)
        elif isinstance(key, str):
            try:
                parsed, reso = self._parse_with_reso(key)
            except (ValueError, pytz.NonExistentTimeError) as err:
                raise KeyError(key) from err
            self._disallow_mismatched_indexing(parsed)
            if self._can_partial_date_slice(reso):
                try:
                    return self._partial_date_slice(reso, parsed)
                except KeyError as err:
                    raise KeyError(key) from err
            key = parsed
        elif isinstance(key, dt.timedelta):
            raise TypeError(f'Cannot index {type(self).__name__} with {type(key).__name__}')
        elif isinstance(key, dt.time):
            return self.indexer_at_time(key)
        else:
            raise KeyError(key)
        try:
            return Index.get_loc(self, key)
        except KeyError as err:
            raise KeyError(orig_key) from err

    @doc(DatetimeTimedeltaMixin._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label, side: str):
        if isinstance(label, dt.date) and (not isinstance(label, dt.datetime)):
            label = Timestamp(label).to_pydatetime()
        label = super()._maybe_cast_slice_bound(label, side)
        self._data._assert_tzawareness_compat(label)
        return Timestamp(label)

    def slice_indexer(self, start=None, end=None, step=None):
        """
        Return indexer for specified label slice.
        Index.slice_indexer, customized to handle time slicing.

        In addition to functionality provided by Index.slice_indexer, does the
        following:

        - if both `start` and `end` are instances of `datetime.time`, it
          invokes `indexer_between_time`
        - if `start` and `end` are both either string or None perform
          value-based selection in non-monotonic cases.

        """
        if isinstance(start, dt.time) and isinstance(end, dt.time):
            if step is not None and step != 1:
                raise ValueError('Must have step size of 1 with time slices')
            return self.indexer_between_time(start, end)
        if isinstance(start, dt.time) or isinstance(end, dt.time):
            raise KeyError('Cannot mix time and non-time slice keys')

        def check_str_or_none(point) -> bool:
            return point is not None and (not isinstance(point, str))
        if check_str_or_none(start) or check_str_or_none(end) or self.is_monotonic_increasing:
            return Index.slice_indexer(self, start, end, step)
        mask = np.array(True)
        in_index = True
        if start is not None:
            start_casted = self._maybe_cast_slice_bound(start, 'left')
            mask = start_casted <= self
            in_index &= (start_casted == self).any()
        if end is not None:
            end_casted = self._maybe_cast_slice_bound(end, 'right')
            mask = (self <= end_casted) & mask
            in_index &= (end_casted == self).any()
        if not in_index:
            raise KeyError('Value based partial slicing on non-monotonic DatetimeIndexes with non-existing keys is not allowed.')
        indexer = mask.nonzero()[0][::step]
        if len(indexer) == len(self):
            return slice(None)
        else:
            return indexer

    @property
    def inferred_type(self) -> str:
        return 'datetime64'

    def indexer_at_time(self, time, asof: bool=False) -> npt.NDArray[np.intp]:
        """
        Return index locations of values at particular time of day.

        Parameters
        ----------
        time : datetime.time or str
            Time passed in either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p", "%I%M%S%p").

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        indexer_between_time : Get index locations of values between particular
            times of day.
        DataFrame.at_time : Select values at particular time of day.

        Examples
        --------
        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00", "2/1/2020 11:00",
        ...                         "3/1/2020 10:00"])
        >>> idx.indexer_at_time("10:00")
        array([0, 2])
        """
        if asof:
            raise NotImplementedError("'asof' argument is not supported")
        if isinstance(time, str):
            from dateutil.parser import parse
            time = parse(time).time()
        if time.tzinfo:
            if self.tz is None:
                raise ValueError('Index must be timezone aware.')
            time_micros = self.tz_convert(time.tzinfo)._get_time_micros()
        else:
            time_micros = self._get_time_micros()
        micros = _time_to_micros(time)
        return (time_micros == micros).nonzero()[0]

    def indexer_between_time(self, start_time, end_time, include_start: bool=True, include_end: bool=True) -> npt.NDArray[np.intp]:
        """
        Return index locations of values between particular times of day.

        Parameters
        ----------
        start_time, end_time : datetime.time, str
            Time passed either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p","%I%M%S%p").
        include_start : bool, default True
        include_end : bool, default True

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        indexer_at_time : Get index locations of values at particular time of day.
        DataFrame.between_time : Select values between particular times of day.

        Examples
        --------
        >>> idx = pd.date_range("2023-01-01", periods=4, freq="h")
        >>> idx
        DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 01:00:00',
                           '2023-01-01 02:00:00', '2023-01-01 03:00:00'],
                          dtype='datetime64[ns]', freq='h')
        >>> idx.indexer_between_time("00:00", "2:00", include_end=False)
        array([0, 1])
        """
        start_time = to_time(start_time)
        end_time = to_time(end_time)
        time_micros = self._get_time_micros()
        start_micros = _time_to_micros(start_time)
        end_micros = _time_to_micros(end_time)
        if include_start and include_end:
            lop = rop = operator.le
        elif include_start:
            lop = operator.le
            rop = operator.lt
        elif include_end:
            lop = operator.lt
            rop = operator.le
        else:
            lop = rop = operator.lt
        if start_time <= end_time:
            join_op = operator.and_
        else:
            join_op = operator.or_
        mask = join_op(lop(start_micros, time_micros), rop(time_micros, end_micros))
        return mask.nonzero()[0]