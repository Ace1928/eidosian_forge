from __future__ import annotations
from datetime import (
from functools import wraps
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas._libs.tslibs.timedeltas import get_unit_for_round
from pandas._libs.tslibs.timestamps import integer_op_not_supported
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.algorithms import (
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import (
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.ops.invalid import (
from pandas.tseries import frequencies
class DatetimeLikeArrayMixin(OpsMixin, NDArrayBackedExtensionArray):
    """
    Shared Base/Mixin class for DatetimeArray, TimedeltaArray, PeriodArray

    Assumes that __new__/__init__ defines:
        _ndarray

    and that inheriting subclass implements:
        freq
    """
    _infer_matches: tuple[str, ...]
    _is_recognized_dtype: Callable[[DtypeObj], bool]
    _recognized_scalars: tuple[type, ...]
    _ndarray: np.ndarray
    freq: BaseOffset | None

    @cache_readonly
    def _can_hold_na(self) -> bool:
        return True

    def __init__(self, data, dtype: Dtype | None=None, freq=None, copy: bool=False) -> None:
        raise AbstractMethodError(self)

    @property
    def _scalar_type(self) -> type[DatetimeLikeScalar]:
        """
        The scalar associated with this datelike

        * PeriodArray : Period
        * DatetimeArray : Timestamp
        * TimedeltaArray : Timedelta
        """
        raise AbstractMethodError(self)

    def _scalar_from_string(self, value: str) -> DTScalarOrNaT:
        """
        Construct a scalar type from a string.

        Parameters
        ----------
        value : str

        Returns
        -------
        Period, Timestamp, or Timedelta, or NaT
            Whatever the type of ``self._scalar_type`` is.

        Notes
        -----
        This should call ``self._check_compatible_with`` before
        unboxing the result.
        """
        raise AbstractMethodError(self)

    def _unbox_scalar(self, value: DTScalarOrNaT) -> np.int64 | np.datetime64 | np.timedelta64:
        """
        Unbox the integer value of a scalar `value`.

        Parameters
        ----------
        value : Period, Timestamp, Timedelta, or NaT
            Depending on subclass.

        Returns
        -------
        int

        Examples
        --------
        >>> arr = pd.array(np.array(['1970-01-01'], 'datetime64[ns]'))
        >>> arr._unbox_scalar(arr[0])
        numpy.datetime64('1970-01-01T00:00:00.000000000')
        """
        raise AbstractMethodError(self)

    def _check_compatible_with(self, other: DTScalarOrNaT) -> None:
        """
        Verify that `self` and `other` are compatible.

        * DatetimeArray verifies that the timezones (if any) match
        * PeriodArray verifies that the freq matches
        * Timedelta has no verification

        In each case, NaT is considered compatible.

        Parameters
        ----------
        other

        Raises
        ------
        Exception
        """
        raise AbstractMethodError(self)

    def _box_func(self, x):
        """
        box function to get object from internal representation
        """
        raise AbstractMethodError(self)

    def _box_values(self, values) -> np.ndarray:
        """
        apply box func to passed values
        """
        return lib.map_infer(values, self._box_func, convert=False)

    def __iter__(self) -> Iterator:
        if self.ndim > 1:
            return (self[n] for n in range(len(self)))
        else:
            return (self._box_func(v) for v in self.asi8)

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        """
        Integer representation of the values.

        Returns
        -------
        ndarray
            An ndarray with int64 dtype.
        """
        return self._ndarray.view('i8')

    def _format_native_types(self, *, na_rep: str | float='NaT', date_format=None) -> npt.NDArray[np.object_]:
        """
        Helper method for astype when converting to strings.

        Returns
        -------
        ndarray[str]
        """
        raise AbstractMethodError(self)

    def _formatter(self, boxed: bool=False):
        return "'{}'".format

    def __array__(self, dtype: NpDtype | None=None) -> np.ndarray:
        if is_object_dtype(dtype):
            return np.array(list(self), dtype=object)
        return self._ndarray

    @overload
    def __getitem__(self, item: ScalarIndexer) -> DTScalarOrNaT:
        ...

    @overload
    def __getitem__(self, item: SequenceIndexer | PositionalIndexerTuple) -> Self:
        ...

    def __getitem__(self, key: PositionalIndexer2D) -> Self | DTScalarOrNaT:
        """
        This getitem defers to the underlying array, which by-definition can
        only handle list-likes, slices, and integer scalars
        """
        result = cast('Union[Self, DTScalarOrNaT]', super().__getitem__(key))
        if lib.is_scalar(result):
            return result
        else:
            result = cast(Self, result)
        result._freq = self._get_getitem_freq(key)
        return result

    def _get_getitem_freq(self, key) -> BaseOffset | None:
        """
        Find the `freq` attribute to assign to the result of a __getitem__ lookup.
        """
        is_period = isinstance(self.dtype, PeriodDtype)
        if is_period:
            freq = self.freq
        elif self.ndim != 1:
            freq = None
        else:
            key = check_array_indexer(self, key)
            freq = None
            if isinstance(key, slice):
                if self.freq is not None and key.step is not None:
                    freq = key.step * self.freq
                else:
                    freq = self.freq
            elif key is Ellipsis:
                freq = self.freq
            elif com.is_bool_indexer(key):
                new_key = lib.maybe_booleans_to_slice(key.view(np.uint8))
                if isinstance(new_key, slice):
                    return self._get_getitem_freq(new_key)
        return freq

    def __setitem__(self, key: int | Sequence[int] | Sequence[bool] | slice, value: NaTType | Any | Sequence[Any]) -> None:
        no_op = check_setitem_lengths(key, value, self)
        super().__setitem__(key, value)
        if no_op:
            return
        self._maybe_clear_freq()

    def _maybe_clear_freq(self) -> None:
        pass

    def astype(self, dtype, copy: bool=True):
        dtype = pandas_dtype(dtype)
        if dtype == object:
            if self.dtype.kind == 'M':
                self = cast('DatetimeArray', self)
                i8data = self.asi8
                converted = ints_to_pydatetime(i8data, tz=self.tz, box='timestamp', reso=self._creso)
                return converted
            elif self.dtype.kind == 'm':
                return ints_to_pytimedelta(self._ndarray, box=True)
            return self._box_values(self.asi8.ravel()).reshape(self.shape)
        elif isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)
        elif is_string_dtype(dtype):
            return self._format_native_types()
        elif dtype.kind in 'iu':
            values = self.asi8
            if dtype != np.int64:
                raise TypeError(f"Converting from {self.dtype} to {dtype} is not supported. Do obj.astype('int64').astype(dtype) instead")
            if copy:
                values = values.copy()
            return values
        elif dtype.kind in 'mM' and self.dtype != dtype or dtype.kind == 'f':
            msg = f'Cannot cast {type(self).__name__} to dtype {dtype}'
            raise TypeError(msg)
        else:
            return np.asarray(self, dtype=dtype)

    @overload
    def view(self) -> Self:
        ...

    @overload
    def view(self, dtype: Literal['M8[ns]']) -> DatetimeArray:
        ...

    @overload
    def view(self, dtype: Literal['m8[ns]']) -> TimedeltaArray:
        ...

    @overload
    def view(self, dtype: Dtype | None=...) -> ArrayLike:
        ...

    def view(self, dtype: Dtype | None=None) -> ArrayLike:
        return super().view(dtype)

    def _validate_comparison_value(self, other):
        if isinstance(other, str):
            try:
                other = self._scalar_from_string(other)
            except (ValueError, IncompatibleFrequency):
                raise InvalidComparison(other)
        if isinstance(other, self._recognized_scalars) or other is NaT:
            other = self._scalar_type(other)
            try:
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                raise InvalidComparison(other) from err
        elif not is_list_like(other):
            raise InvalidComparison(other)
        elif len(other) != len(self):
            raise ValueError('Lengths must match')
        else:
            try:
                other = self._validate_listlike(other, allow_object=True)
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                if is_object_dtype(getattr(other, 'dtype', None)):
                    pass
                else:
                    raise InvalidComparison(other) from err
        return other

    def _validate_scalar(self, value, *, allow_listlike: bool=False, unbox: bool=True):
        """
        Validate that the input value can be cast to our scalar_type.

        Parameters
        ----------
        value : object
        allow_listlike: bool, default False
            When raising an exception, whether the message should say
            listlike inputs are allowed.
        unbox : bool, default True
            Whether to unbox the result before returning.  Note: unbox=False
            skips the setitem compatibility check.

        Returns
        -------
        self._scalar_type or NaT
        """
        if isinstance(value, self._scalar_type):
            pass
        elif isinstance(value, str):
            try:
                value = self._scalar_from_string(value)
            except ValueError as err:
                msg = self._validation_error_message(value, allow_listlike)
                raise TypeError(msg) from err
        elif is_valid_na_for_dtype(value, self.dtype):
            value = NaT
        elif isna(value):
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)
        elif isinstance(value, self._recognized_scalars):
            value = self._scalar_type(value)
        else:
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)
        if not unbox:
            return value
        return self._unbox_scalar(value)

    def _validation_error_message(self, value, allow_listlike: bool=False) -> str:
        """
        Construct an exception message on validation error.

        Some methods allow only scalar inputs, while others allow either scalar
        or listlike.

        Parameters
        ----------
        allow_listlike: bool, default False

        Returns
        -------
        str
        """
        if hasattr(value, 'dtype') and getattr(value, 'ndim', 0) > 0:
            msg_got = f'{value.dtype} array'
        else:
            msg_got = f"'{type(value).__name__}'"
        if allow_listlike:
            msg = f"value should be a '{self._scalar_type.__name__}', 'NaT', or array of those. Got {msg_got} instead."
        else:
            msg = f"value should be a '{self._scalar_type.__name__}' or 'NaT'. Got {msg_got} instead."
        return msg

    def _validate_listlike(self, value, allow_object: bool=False):
        if isinstance(value, type(self)):
            if self.dtype.kind in 'mM' and (not allow_object):
                value = value.as_unit(self.unit, round_ok=False)
            return value
        if isinstance(value, list) and len(value) == 0:
            return type(self)._from_sequence([], dtype=self.dtype)
        if hasattr(value, 'dtype') and value.dtype == object:
            if lib.infer_dtype(value) in self._infer_matches:
                try:
                    value = type(self)._from_sequence(value)
                except (ValueError, TypeError):
                    if allow_object:
                        return value
                    msg = self._validation_error_message(value, True)
                    raise TypeError(msg)
        value = extract_array(value, extract_numpy=True)
        value = pd_array(value)
        value = extract_array(value, extract_numpy=True)
        if is_all_strings(value):
            try:
                value = type(self)._from_sequence(value, dtype=self.dtype)
            except ValueError:
                pass
        if isinstance(value.dtype, CategoricalDtype):
            if value.categories.dtype == self.dtype:
                value = value._internal_get_values()
                value = extract_array(value, extract_numpy=True)
        if allow_object and is_object_dtype(value.dtype):
            pass
        elif not type(self)._is_recognized_dtype(value.dtype):
            msg = self._validation_error_message(value, True)
            raise TypeError(msg)
        if self.dtype.kind in 'mM' and (not allow_object):
            value = value.as_unit(self.unit, round_ok=False)
        return value

    def _validate_setitem_value(self, value):
        if is_list_like(value):
            value = self._validate_listlike(value)
        else:
            return self._validate_scalar(value, allow_listlike=True)
        return self._unbox(value)

    @final
    def _unbox(self, other) -> np.int64 | np.datetime64 | np.timedelta64 | np.ndarray:
        """
        Unbox either a scalar with _unbox_scalar or an instance of our own type.
        """
        if lib.is_scalar(other):
            other = self._unbox_scalar(other)
        else:
            self._check_compatible_with(other)
            other = other._ndarray
        return other

    @ravel_compat
    def map(self, mapper, na_action=None):
        from pandas import Index
        result = map_array(self, mapper, na_action=na_action)
        result = Index(result)
        if isinstance(result, ABCMultiIndex):
            return result.to_numpy()
        else:
            return result.array

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        """
        Compute boolean array of whether each value is found in the
        passed set of values.

        Parameters
        ----------
        values : np.ndarray or ExtensionArray

        Returns
        -------
        ndarray[bool]
        """
        if values.dtype.kind in 'fiuc':
            return np.zeros(self.shape, dtype=bool)
        values = ensure_wrapped_if_datetimelike(values)
        if not isinstance(values, type(self)):
            inferable = ['timedelta', 'timedelta64', 'datetime', 'datetime64', 'date', 'period']
            if values.dtype == object:
                values = lib.maybe_convert_objects(values, convert_non_numeric=True, dtype_if_all_nat=self.dtype)
                if values.dtype != object:
                    return self.isin(values)
                inferred = lib.infer_dtype(values, skipna=False)
                if inferred not in inferable:
                    if inferred == 'string':
                        pass
                    elif 'mixed' in inferred:
                        return isin(self.astype(object), values)
                    else:
                        return np.zeros(self.shape, dtype=bool)
            try:
                values = type(self)._from_sequence(values)
            except ValueError:
                return isin(self.astype(object), values)
            else:
                warnings.warn(f"The behavior of 'isin' with dtype={self.dtype} and castable values (e.g. strings) is deprecated. In a future version, these will not be considered matching by isin. Explicitly cast to the appropriate dtype before calling isin instead.", FutureWarning, stacklevel=find_stack_level())
        if self.dtype.kind in 'mM':
            self = cast('DatetimeArray | TimedeltaArray', self)
            values = values.as_unit(self.unit)
        try:
            self._check_compatible_with(values)
        except (TypeError, ValueError):
            return np.zeros(self.shape, dtype=bool)
        return isin(self.asi8, values.asi8)

    def isna(self) -> npt.NDArray[np.bool_]:
        return self._isnan

    @property
    def _isnan(self) -> npt.NDArray[np.bool_]:
        """
        return if each value is nan
        """
        return self.asi8 == iNaT

    @property
    def _hasna(self) -> bool:
        """
        return if I have any nans; enables various perf speedups
        """
        return bool(self._isnan.any())

    def _maybe_mask_results(self, result: np.ndarray, fill_value=iNaT, convert=None) -> np.ndarray:
        """
        Parameters
        ----------
        result : np.ndarray
        fill_value : object, default iNaT
        convert : str, dtype or None

        Returns
        -------
        result : ndarray with values replace by the fill_value

        mask the result if needed, convert to the provided dtype if its not
        None

        This is an internal routine.
        """
        if self._hasna:
            if convert:
                result = result.astype(convert)
            if fill_value is None:
                fill_value = np.nan
            np.putmask(result, self._isnan, fill_value)
        return result

    @property
    def freqstr(self) -> str | None:
        """
        Return the frequency object as a string if it's set, otherwise None.

        Examples
        --------
        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00"], freq="D")
        >>> idx.freqstr
        'D'

        The frequency can be inferred if there are more than 2 points:

        >>> idx = pd.DatetimeIndex(["2018-01-01", "2018-01-03", "2018-01-05"],
        ...                        freq="infer")
        >>> idx.freqstr
        '2D'

        For PeriodIndex:

        >>> idx = pd.PeriodIndex(["2023-1", "2023-2", "2023-3"], freq="M")
        >>> idx.freqstr
        'M'
        """
        if self.freq is None:
            return None
        return self.freq.freqstr

    @property
    def inferred_freq(self) -> str | None:
        """
        Tries to return a string representing a frequency generated by infer_freq.

        Returns None if it can't autodetect the frequency.

        Examples
        --------
        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["2018-01-01", "2018-01-03", "2018-01-05"])
        >>> idx.inferred_freq
        '2D'

        For TimedeltaIndex:

        >>> tdelta_idx = pd.to_timedelta(["0 days", "10 days", "20 days"])
        >>> tdelta_idx
        TimedeltaIndex(['0 days', '10 days', '20 days'],
                       dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.inferred_freq
        '10D'
        """
        if self.ndim != 1:
            return None
        try:
            return frequencies.infer_freq(self)
        except ValueError:
            return None

    @property
    def _resolution_obj(self) -> Resolution | None:
        freqstr = self.freqstr
        if freqstr is None:
            return None
        try:
            return Resolution.get_reso_from_freqstr(freqstr)
        except KeyError:
            return None

    @property
    def resolution(self) -> str:
        """
        Returns day, hour, minute, second, millisecond or microsecond
        """
        return self._resolution_obj.attrname

    @property
    def _is_monotonic_increasing(self) -> bool:
        return algos.is_monotonic(self.asi8, timelike=True)[0]

    @property
    def _is_monotonic_decreasing(self) -> bool:
        return algos.is_monotonic(self.asi8, timelike=True)[1]

    @property
    def _is_unique(self) -> bool:
        return len(unique1d(self.asi8.ravel('K'))) == self.size

    def _cmp_method(self, other, op):
        if self.ndim > 1 and getattr(other, 'shape', None) == self.shape:
            return op(self.ravel(), other.ravel()).reshape(self.shape)
        try:
            other = self._validate_comparison_value(other)
        except InvalidComparison:
            return invalid_comparison(self, other, op)
        dtype = getattr(other, 'dtype', None)
        if is_object_dtype(dtype):
            result = ops.comp_method_OBJECT_ARRAY(op, np.asarray(self.astype(object)), other)
            return result
        if other is NaT:
            if op is operator.ne:
                result = np.ones(self.shape, dtype=bool)
            else:
                result = np.zeros(self.shape, dtype=bool)
            return result
        if not isinstance(self.dtype, PeriodDtype):
            self = cast(TimelikeOps, self)
            if self._creso != other._creso:
                if not isinstance(other, type(self)):
                    try:
                        other = other.as_unit(self.unit, round_ok=False)
                    except ValueError:
                        other_arr = np.array(other.asm8)
                        return compare_mismatched_resolutions(self._ndarray, other_arr, op)
                else:
                    other_arr = other._ndarray
                    return compare_mismatched_resolutions(self._ndarray, other_arr, op)
        other_vals = self._unbox(other)
        result = op(self._ndarray.view('i8'), other_vals.view('i8'))
        o_mask = isna(other)
        mask = self._isnan | o_mask
        if mask.any():
            nat_result = op is operator.ne
            np.putmask(result, mask, nat_result)
        return result
    __pow__ = _make_unpacked_invalid_op('__pow__')
    __rpow__ = _make_unpacked_invalid_op('__rpow__')
    __mul__ = _make_unpacked_invalid_op('__mul__')
    __rmul__ = _make_unpacked_invalid_op('__rmul__')
    __truediv__ = _make_unpacked_invalid_op('__truediv__')
    __rtruediv__ = _make_unpacked_invalid_op('__rtruediv__')
    __floordiv__ = _make_unpacked_invalid_op('__floordiv__')
    __rfloordiv__ = _make_unpacked_invalid_op('__rfloordiv__')
    __mod__ = _make_unpacked_invalid_op('__mod__')
    __rmod__ = _make_unpacked_invalid_op('__rmod__')
    __divmod__ = _make_unpacked_invalid_op('__divmod__')
    __rdivmod__ = _make_unpacked_invalid_op('__rdivmod__')

    @final
    def _get_i8_values_and_mask(self, other) -> tuple[int | npt.NDArray[np.int64], None | npt.NDArray[np.bool_]]:
        """
        Get the int64 values and b_mask to pass to add_overflowsafe.
        """
        if isinstance(other, Period):
            i8values = other.ordinal
            mask = None
        elif isinstance(other, (Timestamp, Timedelta)):
            i8values = other._value
            mask = None
        else:
            mask = other._isnan
            i8values = other.asi8
        return (i8values, mask)

    @final
    def _get_arithmetic_result_freq(self, other) -> BaseOffset | None:
        """
        Check if we can preserve self.freq in addition or subtraction.
        """
        if isinstance(self.dtype, PeriodDtype):
            return self.freq
        elif not lib.is_scalar(other):
            return None
        elif isinstance(self.freq, Tick):
            return self.freq
        return None

    @final
    def _add_datetimelike_scalar(self, other) -> DatetimeArray:
        if not lib.is_np_dtype(self.dtype, 'm'):
            raise TypeError(f'cannot add {type(self).__name__} and {type(other).__name__}')
        self = cast('TimedeltaArray', self)
        from pandas.core.arrays import DatetimeArray
        from pandas.core.arrays.datetimes import tz_to_dtype
        assert other is not NaT
        if isna(other):
            result = self._ndarray + NaT.to_datetime64().astype(f'M8[{self.unit}]')
            return DatetimeArray._simple_new(result, dtype=result.dtype)
        other = Timestamp(other)
        self, other = self._ensure_matching_resos(other)
        self = cast('TimedeltaArray', self)
        other_i8, o_mask = self._get_i8_values_and_mask(other)
        result = add_overflowsafe(self.asi8, np.asarray(other_i8, dtype='i8'))
        res_values = result.view(f'M8[{self.unit}]')
        dtype = tz_to_dtype(tz=other.tz, unit=self.unit)
        res_values = result.view(f'M8[{self.unit}]')
        new_freq = self._get_arithmetic_result_freq(other)
        return DatetimeArray._simple_new(res_values, dtype=dtype, freq=new_freq)

    @final
    def _add_datetime_arraylike(self, other: DatetimeArray) -> DatetimeArray:
        if not lib.is_np_dtype(self.dtype, 'm'):
            raise TypeError(f'cannot add {type(self).__name__} and {type(other).__name__}')
        return other + self

    @final
    def _sub_datetimelike_scalar(self, other: datetime | np.datetime64) -> TimedeltaArray:
        if self.dtype.kind != 'M':
            raise TypeError(f'cannot subtract a datelike from a {type(self).__name__}')
        self = cast('DatetimeArray', self)
        if isna(other):
            return self - NaT
        ts = Timestamp(other)
        self, ts = self._ensure_matching_resos(ts)
        return self._sub_datetimelike(ts)

    @final
    def _sub_datetime_arraylike(self, other: DatetimeArray) -> TimedeltaArray:
        if self.dtype.kind != 'M':
            raise TypeError(f'cannot subtract a datelike from a {type(self).__name__}')
        if len(self) != len(other):
            raise ValueError('cannot add indices of unequal length')
        self = cast('DatetimeArray', self)
        self, other = self._ensure_matching_resos(other)
        return self._sub_datetimelike(other)

    @final
    def _sub_datetimelike(self, other: Timestamp | DatetimeArray) -> TimedeltaArray:
        self = cast('DatetimeArray', self)
        from pandas.core.arrays import TimedeltaArray
        try:
            self._assert_tzawareness_compat(other)
        except TypeError as err:
            new_message = str(err).replace('compare', 'subtract')
            raise type(err)(new_message) from err
        other_i8, o_mask = self._get_i8_values_and_mask(other)
        res_values = add_overflowsafe(self.asi8, np.asarray(-other_i8, dtype='i8'))
        res_m8 = res_values.view(f'timedelta64[{self.unit}]')
        new_freq = self._get_arithmetic_result_freq(other)
        new_freq = cast('Tick | None', new_freq)
        return TimedeltaArray._simple_new(res_m8, dtype=res_m8.dtype, freq=new_freq)

    @final
    def _add_period(self, other: Period) -> PeriodArray:
        if not lib.is_np_dtype(self.dtype, 'm'):
            raise TypeError(f'cannot add Period to a {type(self).__name__}')
        from pandas.core.arrays.period import PeriodArray
        i8vals = np.broadcast_to(other.ordinal, self.shape)
        dtype = PeriodDtype(other.freq)
        parr = PeriodArray(i8vals, dtype=dtype)
        return parr + self

    def _add_offset(self, offset):
        raise AbstractMethodError(self)

    def _add_timedeltalike_scalar(self, other):
        """
        Add a delta of a timedeltalike

        Returns
        -------
        Same type as self
        """
        if isna(other):
            new_values = np.empty(self.shape, dtype='i8').view(self._ndarray.dtype)
            new_values.fill(iNaT)
            return type(self)._simple_new(new_values, dtype=self.dtype)
        self = cast('DatetimeArray | TimedeltaArray', self)
        other = Timedelta(other)
        self, other = self._ensure_matching_resos(other)
        return self._add_timedeltalike(other)

    def _add_timedelta_arraylike(self, other: TimedeltaArray):
        """
        Add a delta of a TimedeltaIndex

        Returns
        -------
        Same type as self
        """
        if len(self) != len(other):
            raise ValueError('cannot add indices of unequal length')
        self = cast('DatetimeArray | TimedeltaArray', self)
        self, other = self._ensure_matching_resos(other)
        return self._add_timedeltalike(other)

    @final
    def _add_timedeltalike(self, other: Timedelta | TimedeltaArray):
        self = cast('DatetimeArray | TimedeltaArray', self)
        other_i8, o_mask = self._get_i8_values_and_mask(other)
        new_values = add_overflowsafe(self.asi8, np.asarray(other_i8, dtype='i8'))
        res_values = new_values.view(self._ndarray.dtype)
        new_freq = self._get_arithmetic_result_freq(other)
        return type(self)._simple_new(res_values, dtype=self.dtype, freq=new_freq)

    @final
    def _add_nat(self):
        """
        Add pd.NaT to self
        """
        if isinstance(self.dtype, PeriodDtype):
            raise TypeError(f'Cannot add {type(self).__name__} and {type(NaT).__name__}')
        self = cast('TimedeltaArray | DatetimeArray', self)
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        result = result.view(self._ndarray.dtype)
        return type(self)._simple_new(result, dtype=self.dtype, freq=None)

    @final
    def _sub_nat(self):
        """
        Subtract pd.NaT from self
        """
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        if self.dtype.kind in 'mM':
            self = cast('DatetimeArray| TimedeltaArray', self)
            return result.view(f'timedelta64[{self.unit}]')
        else:
            return result.view('timedelta64[ns]')

    @final
    def _sub_periodlike(self, other: Period | PeriodArray) -> npt.NDArray[np.object_]:
        if not isinstance(self.dtype, PeriodDtype):
            raise TypeError(f'cannot subtract {type(other).__name__} from {type(self).__name__}')
        self = cast('PeriodArray', self)
        self._check_compatible_with(other)
        other_i8, o_mask = self._get_i8_values_and_mask(other)
        new_i8_data = add_overflowsafe(self.asi8, np.asarray(-other_i8, dtype='i8'))
        new_data = np.array([self.freq.base * x for x in new_i8_data])
        if o_mask is None:
            mask = self._isnan
        else:
            mask = self._isnan | o_mask
        new_data[mask] = NaT
        return new_data

    @final
    def _addsub_object_array(self, other: npt.NDArray[np.object_], op):
        """
        Add or subtract array-like of DateOffset objects

        Parameters
        ----------
        other : np.ndarray[object]
        op : {operator.add, operator.sub}

        Returns
        -------
        np.ndarray[object]
            Except in fastpath case with length 1 where we operate on the
            contained scalar.
        """
        assert op in [operator.add, operator.sub]
        if len(other) == 1 and self.ndim == 1:
            return op(self, other[0])
        warnings.warn(f'Adding/subtracting object-dtype array to {type(self).__name__} not vectorized.', PerformanceWarning, stacklevel=find_stack_level())
        assert self.shape == other.shape, (self.shape, other.shape)
        res_values = op(self.astype('O'), np.asarray(other))
        return res_values

    def _accumulate(self, name: str, *, skipna: bool=True, **kwargs) -> Self:
        if name not in {'cummin', 'cummax'}:
            raise TypeError(f'Accumulation {name} not supported for {type(self)}')
        op = getattr(datetimelike_accumulations, name)
        result = op(self.copy(), skipna=skipna, **kwargs)
        return type(self)._simple_new(result, dtype=self.dtype)

    @unpack_zerodim_and_defer('__add__')
    def __add__(self, other):
        other_dtype = getattr(other, 'dtype', None)
        other = ensure_wrapped_if_datetimelike(other)
        if other is NaT:
            result = self._add_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar(other)
        elif isinstance(other, BaseOffset):
            result = self._add_offset(other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._add_datetimelike_scalar(other)
        elif isinstance(other, Period) and lib.is_np_dtype(self.dtype, 'm'):
            result = self._add_period(other)
        elif lib.is_integer(other):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast('PeriodArray', self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.add)
        elif lib.is_np_dtype(other_dtype, 'm'):
            result = self._add_timedelta_arraylike(other)
        elif is_object_dtype(other_dtype):
            result = self._addsub_object_array(other, operator.add)
        elif lib.is_np_dtype(other_dtype, 'M') or isinstance(other_dtype, DatetimeTZDtype):
            return self._add_datetime_arraylike(other)
        elif is_integer_dtype(other_dtype):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast('PeriodArray', self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.add)
        else:
            return NotImplemented
        if isinstance(result, np.ndarray) and lib.is_np_dtype(result.dtype, 'm'):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray._from_sequence(result)
        return result

    def __radd__(self, other):
        return self.__add__(other)

    @unpack_zerodim_and_defer('__sub__')
    def __sub__(self, other):
        other_dtype = getattr(other, 'dtype', None)
        other = ensure_wrapped_if_datetimelike(other)
        if other is NaT:
            result = self._sub_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar(-other)
        elif isinstance(other, BaseOffset):
            result = self._add_offset(-other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._sub_datetimelike_scalar(other)
        elif lib.is_integer(other):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast('PeriodArray', self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.sub)
        elif isinstance(other, Period):
            result = self._sub_periodlike(other)
        elif lib.is_np_dtype(other_dtype, 'm'):
            result = self._add_timedelta_arraylike(-other)
        elif is_object_dtype(other_dtype):
            result = self._addsub_object_array(other, operator.sub)
        elif lib.is_np_dtype(other_dtype, 'M') or isinstance(other_dtype, DatetimeTZDtype):
            result = self._sub_datetime_arraylike(other)
        elif isinstance(other_dtype, PeriodDtype):
            result = self._sub_periodlike(other)
        elif is_integer_dtype(other_dtype):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast('PeriodArray', self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.sub)
        else:
            return NotImplemented
        if isinstance(result, np.ndarray) and lib.is_np_dtype(result.dtype, 'm'):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray._from_sequence(result)
        return result

    def __rsub__(self, other):
        other_dtype = getattr(other, 'dtype', None)
        other_is_dt64 = lib.is_np_dtype(other_dtype, 'M') or isinstance(other_dtype, DatetimeTZDtype)
        if other_is_dt64 and lib.is_np_dtype(self.dtype, 'm'):
            if lib.is_scalar(other):
                return Timestamp(other) - self
            if not isinstance(other, DatetimeLikeArrayMixin):
                from pandas.core.arrays import DatetimeArray
                other = DatetimeArray._from_sequence(other)
            return other - self
        elif self.dtype.kind == 'M' and hasattr(other, 'dtype') and (not other_is_dt64):
            raise TypeError(f'cannot subtract {type(self).__name__} from {type(other).__name__}')
        elif isinstance(self.dtype, PeriodDtype) and lib.is_np_dtype(other_dtype, 'm'):
            raise TypeError(f'cannot subtract {type(self).__name__} from {other.dtype}')
        elif lib.is_np_dtype(self.dtype, 'm'):
            self = cast('TimedeltaArray', self)
            return -self + other
        return -(self - other)

    def __iadd__(self, other) -> Self:
        result = self + other
        self[:] = result[:]
        if not isinstance(self.dtype, PeriodDtype):
            self._freq = result.freq
        return self

    def __isub__(self, other) -> Self:
        result = self - other
        self[:] = result[:]
        if not isinstance(self.dtype, PeriodDtype):
            self._freq = result.freq
        return self

    @_period_dispatch
    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> Self:
        return super()._quantile(qs=qs, interpolation=interpolation)

    @_period_dispatch
    def min(self, *, axis: AxisInt | None=None, skipna: bool=True, **kwargs):
        """
        Return the minimum value of the Array or minimum along
        an axis.

        See Also
        --------
        numpy.ndarray.min
        Index.min : Return the minimum value in an Index.
        Series.min : Return the minimum value in a Series.
        """
        nv.validate_min((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)
        result = nanops.nanmin(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    @_period_dispatch
    def max(self, *, axis: AxisInt | None=None, skipna: bool=True, **kwargs):
        """
        Return the maximum value of the Array or maximum along
        an axis.

        See Also
        --------
        numpy.ndarray.max
        Index.max : Return the maximum value in an Index.
        Series.max : Return the maximum value in a Series.
        """
        nv.validate_max((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)
        result = nanops.nanmax(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def mean(self, *, skipna: bool=True, axis: AxisInt | None=0):
        """
        Return the mean value of the Array.

        Parameters
        ----------
        skipna : bool, default True
            Whether to ignore any NaT elements.
        axis : int, optional, default 0

        Returns
        -------
        scalar
            Timestamp or Timedelta.

        See Also
        --------
        numpy.ndarray.mean : Returns the average of array elements along a given axis.
        Series.mean : Return the mean value in a Series.

        Notes
        -----
        mean is only defined for Datetime and Timedelta dtypes, not for Period.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.date_range('2001-01-01 00:00', periods=3)
        >>> idx
        DatetimeIndex(['2001-01-01', '2001-01-02', '2001-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.mean()
        Timestamp('2001-01-02 00:00:00')

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='D')
        >>> tdelta_idx
        TimedeltaIndex(['1 days', '2 days', '3 days'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.mean()
        Timedelta('2 days 00:00:00')
        """
        if isinstance(self.dtype, PeriodDtype):
            raise TypeError(f"mean is not implemented for {type(self).__name__} since the meaning is ambiguous.  An alternative is obj.to_timestamp(how='start').mean()")
        result = nanops.nanmean(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())
        return self._wrap_reduction_result(axis, result)

    @_period_dispatch
    def median(self, *, axis: AxisInt | None=None, skipna: bool=True, **kwargs):
        nv.validate_median((), kwargs)
        if axis is not None and abs(axis) >= self.ndim:
            raise ValueError('abs(axis) must be less than ndim')
        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def _mode(self, dropna: bool=True):
        mask = None
        if dropna:
            mask = self.isna()
        i8modes = algorithms.mode(self.view('i8'), mask=mask)
        npmodes = i8modes.view(self._ndarray.dtype)
        npmodes = cast(np.ndarray, npmodes)
        return self._from_backing_data(npmodes)

    def _groupby_op(self, *, how: str, has_dropped_na: bool, min_count: int, ngroups: int, ids: npt.NDArray[np.intp], **kwargs):
        dtype = self.dtype
        if dtype.kind == 'M':
            if how in ['sum', 'prod', 'cumsum', 'cumprod', 'var', 'skew']:
                raise TypeError(f'datetime64 type does not support {how} operations')
            if how in ['any', 'all']:
                warnings.warn(f"'{how}' with datetime64 dtypes is deprecated and will raise in a future version. Use (obj != pd.Timestamp(0)).{how}() instead.", FutureWarning, stacklevel=find_stack_level())
        elif isinstance(dtype, PeriodDtype):
            if how in ['sum', 'prod', 'cumsum', 'cumprod', 'var', 'skew']:
                raise TypeError(f'Period type does not support {how} operations')
            if how in ['any', 'all']:
                warnings.warn(f"'{how}' with PeriodDtype is deprecated and will raise in a future version. Use (obj != pd.Period(0, freq)).{how}() instead.", FutureWarning, stacklevel=find_stack_level())
        elif how in ['prod', 'cumprod', 'skew', 'var']:
            raise TypeError(f'timedelta64 type does not support {how} operations')
        npvalues = self._ndarray.view('M8[ns]')
        from pandas.core.groupby.ops import WrappedCythonOp
        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)
        res_values = op._cython_op_ndim_compat(npvalues, min_count=min_count, ngroups=ngroups, comp_ids=ids, mask=None, **kwargs)
        if op.how in op.cast_blocklist:
            return res_values
        assert res_values.dtype == 'M8[ns]'
        if how in ['std', 'sem']:
            from pandas.core.arrays import TimedeltaArray
            if isinstance(self.dtype, PeriodDtype):
                raise TypeError("'std' and 'sem' are not valid for PeriodDtype")
            self = cast('DatetimeArray | TimedeltaArray', self)
            new_dtype = f'm8[{self.unit}]'
            res_values = res_values.view(new_dtype)
            return TimedeltaArray._simple_new(res_values, dtype=res_values.dtype)
        res_values = res_values.view(self._ndarray.dtype)
        return self._from_backing_data(res_values)