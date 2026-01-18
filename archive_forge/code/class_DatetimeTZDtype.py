from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
from typing import (
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.util import capitalize_first_letter
@register_extension_dtype
class DatetimeTZDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for timezone-aware datetime data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    unit : str, default "ns"
        The precision of the datetime data. Currently limited
        to ``"ns"``.
    tz : str, int, or datetime.tzinfo
        The timezone.

    Attributes
    ----------
    unit
    tz

    Methods
    -------
    None

    Raises
    ------
    ZoneInfoNotFoundError
        When the requested timezone cannot be found.

    Examples
    --------
    >>> from zoneinfo import ZoneInfo
    >>> pd.DatetimeTZDtype(tz=ZoneInfo('UTC'))
    datetime64[ns, UTC]

    >>> pd.DatetimeTZDtype(tz=ZoneInfo('Europe/Paris'))
    datetime64[ns, Europe/Paris]
    """
    type: type[Timestamp] = Timestamp
    kind: str_type = 'M'
    num = 101
    _metadata = ('unit', 'tz')
    _match = re.compile('(datetime64|M8)\\[(?P<unit>.+), (?P<tz>.+)\\]')
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _supports_2d = True
    _can_fast_transpose = True

    @property
    def na_value(self) -> NaTType:
        return NaT

    @cache_readonly
    def base(self) -> DtypeObj:
        return np.dtype(f'M8[{self.unit}]')

    @cache_readonly
    def str(self) -> str:
        return f'|M8[{self.unit}]'

    def __init__(self, unit: str_type | DatetimeTZDtype='ns', tz=None) -> None:
        if isinstance(unit, DatetimeTZDtype):
            unit, tz = (unit.unit, unit.tz)
        if unit != 'ns':
            if isinstance(unit, str) and tz is None:
                result = type(self).construct_from_string(unit)
                unit = result.unit
                tz = result.tz
                msg = f"Passing a dtype alias like 'datetime64[ns, {tz}]' to DatetimeTZDtype is no longer supported. Use 'DatetimeTZDtype.construct_from_string()' instead."
                raise ValueError(msg)
            if unit not in ['s', 'ms', 'us', 'ns']:
                raise ValueError('DatetimeTZDtype only supports s, ms, us, ns units')
        if tz:
            tz = timezones.maybe_get_tz(tz)
            tz = timezones.tz_standardize(tz)
        elif tz is not None:
            raise pytz.UnknownTimeZoneError(tz)
        if tz is None:
            raise TypeError("A 'tz' is required.")
        self._unit = unit
        self._tz = tz

    @cache_readonly
    def _creso(self) -> int:
        """
        The NPY_DATETIMEUNIT corresponding to this dtype's resolution.
        """
        return abbrev_to_npy_unit(self.unit)

    @property
    def unit(self) -> str_type:
        """
        The precision of the datetime data.

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo('America/Los_Angeles'))
        >>> dtype.unit
        'ns'
        """
        return self._unit

    @property
    def tz(self) -> tzinfo:
        """
        The timezone.

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo('America/Los_Angeles'))
        >>> dtype.tz
        zoneinfo.ZoneInfo(key='America/Los_Angeles')
        """
        return self._tz

    @classmethod
    def construct_array_type(cls) -> type_t[DatetimeArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import DatetimeArray
        return DatetimeArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> DatetimeTZDtype:
        """
        Construct a DatetimeTZDtype from a string.

        Parameters
        ----------
        string : str
            The string alias for this DatetimeTZDtype.
            Should be formatted like ``datetime64[ns, <tz>]``,
            where ``<tz>`` is the timezone name.

        Examples
        --------
        >>> DatetimeTZDtype.construct_from_string('datetime64[ns, UTC]')
        datetime64[ns, UTC]
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        match = cls._match.match(string)
        if match:
            d = match.groupdict()
            try:
                return cls(unit=d['unit'], tz=d['tz'])
            except (KeyError, TypeError, ValueError) as err:
                raise TypeError(msg) from err
        raise TypeError(msg)

    def __str__(self) -> str_type:
        return f'datetime64[{self.unit}, {self.tz}]'

    @property
    def name(self) -> str_type:
        """A string representation of the dtype."""
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            if other.startswith('M8['):
                other = f'datetime64[{other[3:]}'
            return other == self.name
        return isinstance(other, DatetimeTZDtype) and self.unit == other.unit and tz_compare(self.tz, other.tz)

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> DatetimeArray:
        """
        Construct DatetimeArray from pyarrow Array/ChunkedArray.

        Note: If the units in the pyarrow Array are the same as this
        DatetimeDtype, then values corresponding to the integer representation
        of ``NaT`` (e.g. one nanosecond before :attr:`pandas.Timestamp.min`)
        are converted to ``NaT``, regardless of the null indicator in the
        pyarrow array.

        Parameters
        ----------
        array : pyarrow.Array or pyarrow.ChunkedArray
            The Arrow array to convert to DatetimeArray.

        Returns
        -------
        extension array : DatetimeArray
        """
        import pyarrow
        from pandas.core.arrays import DatetimeArray
        array = array.cast(pyarrow.timestamp(unit=self._unit), safe=True)
        if isinstance(array, pyarrow.Array):
            np_arr = array.to_numpy(zero_copy_only=False)
        else:
            np_arr = array.to_numpy()
        return DatetimeArray._simple_new(np_arr, dtype=self)

    def __setstate__(self, state) -> None:
        self._tz = state['tz']
        self._unit = state['unit']

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        if all((isinstance(t, DatetimeTZDtype) and t.tz == self.tz for t in dtypes)):
            np_dtype = np.max([cast(DatetimeTZDtype, t).base for t in [self, *dtypes]])
            unit = np.datetime_data(np_dtype)[0]
            return type(self)(unit=unit, tz=self.tz)
        return super()._get_common_dtype(dtypes)

    @cache_readonly
    def index_class(self) -> type_t[DatetimeIndex]:
        from pandas import DatetimeIndex
        return DatetimeIndex