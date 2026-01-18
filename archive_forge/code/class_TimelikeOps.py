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
class TimelikeOps(DatetimeLikeArrayMixin):
    """
    Common ops for TimedeltaIndex/DatetimeIndex, but not PeriodIndex.
    """
    _default_dtype: np.dtype

    def __init__(self, values, dtype=None, freq=lib.no_default, copy: bool=False) -> None:
        warnings.warn(f'{type(self).__name__}.__init__ is deprecated and will be removed in a future version. Use pd.array instead.', FutureWarning, stacklevel=find_stack_level())
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        values = extract_array(values, extract_numpy=True)
        if isinstance(values, IntegerArray):
            values = values.to_numpy('int64', na_value=iNaT)
        inferred_freq = getattr(values, '_freq', None)
        explicit_none = freq is None
        freq = freq if freq is not lib.no_default else None
        if isinstance(values, type(self)):
            if explicit_none:
                pass
            elif freq is None:
                freq = values.freq
            elif freq and values.freq:
                freq = to_offset(freq)
                freq = _validate_inferred_freq(freq, values.freq)
            if dtype is not None and dtype != values.dtype:
                raise TypeError(f'dtype={dtype} does not match data dtype {values.dtype}')
            dtype = values.dtype
            values = values._ndarray
        elif dtype is None:
            if isinstance(values, np.ndarray) and values.dtype.kind in 'Mm':
                dtype = values.dtype
            else:
                dtype = self._default_dtype
                if isinstance(values, np.ndarray) and values.dtype == 'i8':
                    values = values.view(dtype)
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Unexpected type '{type(values).__name__}'. 'values' must be a {type(self).__name__}, ndarray, or Series or Index containing one of those.")
        if values.ndim not in [1, 2]:
            raise ValueError('Only 1-dimensional input arrays are supported.')
        if values.dtype == 'i8':
            if dtype is None:
                dtype = self._default_dtype
                values = values.view(self._default_dtype)
            elif lib.is_np_dtype(dtype, 'mM'):
                values = values.view(dtype)
            elif isinstance(dtype, DatetimeTZDtype):
                kind = self._default_dtype.kind
                new_dtype = f'{kind}8[{dtype.unit}]'
                values = values.view(new_dtype)
        dtype = self._validate_dtype(values, dtype)
        if freq == 'infer':
            raise ValueError(f"Frequency inference not allowed in {type(self).__name__}.__init__. Use 'pd.array()' instead.")
        if copy:
            values = values.copy()
        if freq:
            freq = to_offset(freq)
            if values.dtype.kind == 'm' and (not isinstance(freq, Tick)):
                raise TypeError('TimedeltaArray/Index freq must be a Tick')
        NDArrayBacked.__init__(self, values=values, dtype=dtype)
        self._freq = freq
        if inferred_freq is None and freq is not None:
            type(self)._validate_frequency(self, freq)

    @classmethod
    def _validate_dtype(cls, values, dtype):
        raise AbstractMethodError(cls)

    @property
    def freq(self):
        """
        Return the frequency object if it is set, otherwise None.
        """
        return self._freq

    @freq.setter
    def freq(self, value) -> None:
        if value is not None:
            value = to_offset(value)
            self._validate_frequency(self, value)
            if self.dtype.kind == 'm' and (not isinstance(value, Tick)):
                raise TypeError('TimedeltaArray/Index freq must be a Tick')
            if self.ndim > 1:
                raise ValueError('Cannot set freq with ndim > 1')
        self._freq = value

    @final
    def _maybe_pin_freq(self, freq, validate_kwds: dict):
        """
        Constructor helper to pin the appropriate `freq` attribute.  Assumes
        that self._freq is currently set to any freq inferred in
        _from_sequence_not_strict.
        """
        if freq is None:
            self._freq = None
        elif freq == 'infer':
            if self._freq is None:
                self._freq = to_offset(self.inferred_freq)
        elif freq is lib.no_default:
            pass
        elif self._freq is None:
            freq = to_offset(freq)
            type(self)._validate_frequency(self, freq, **validate_kwds)
            self._freq = freq
        else:
            freq = to_offset(freq)
            _validate_inferred_freq(freq, self._freq)

    @final
    @classmethod
    def _validate_frequency(cls, index, freq: BaseOffset, **kwargs):
        """
        Validate that a frequency is compatible with the values of a given
        Datetime Array/Index or Timedelta Array/Index

        Parameters
        ----------
        index : DatetimeIndex or TimedeltaIndex
            The index on which to determine if the given frequency is valid
        freq : DateOffset
            The frequency to validate
        """
        inferred = index.inferred_freq
        if index.size == 0 or inferred == freq.freqstr:
            return None
        try:
            on_freq = cls._generate_range(start=index[0], end=None, periods=len(index), freq=freq, unit=index.unit, **kwargs)
            if not np.array_equal(index.asi8, on_freq.asi8):
                raise ValueError
        except ValueError as err:
            if 'non-fixed' in str(err):
                raise err
            raise ValueError(f'Inferred frequency {inferred} from passed values does not conform to passed frequency {freq.freqstr}') from err

    @classmethod
    def _generate_range(cls, start, end, periods: int | None, freq, *args, **kwargs) -> Self:
        raise AbstractMethodError(cls)

    @cache_readonly
    def _creso(self) -> int:
        return get_unit_from_dtype(self._ndarray.dtype)

    @cache_readonly
    def unit(self) -> str:
        return dtype_to_unit(self.dtype)

    def as_unit(self, unit: str, round_ok: bool=True) -> Self:
        if unit not in ['s', 'ms', 'us', 'ns']:
            raise ValueError("Supported units are 's', 'ms', 'us', 'ns'")
        dtype = np.dtype(f'{self.dtype.kind}8[{unit}]')
        new_values = astype_overflowsafe(self._ndarray, dtype, round_ok=round_ok)
        if isinstance(self.dtype, np.dtype):
            new_dtype = new_values.dtype
        else:
            tz = cast('DatetimeArray', self).tz
            new_dtype = DatetimeTZDtype(tz=tz, unit=unit)
        return type(self)._simple_new(new_values, dtype=new_dtype, freq=self.freq)

    def _ensure_matching_resos(self, other):
        if self._creso != other._creso:
            if self._creso < other._creso:
                self = self.as_unit(other.unit)
            else:
                other = other.as_unit(self.unit)
        return (self, other)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if ufunc in [np.isnan, np.isinf, np.isfinite] and len(inputs) == 1 and (inputs[0] is self):
            return getattr(ufunc, method)(self._ndarray, **kwargs)
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def _round(self, freq, mode, ambiguous, nonexistent):
        if isinstance(self.dtype, DatetimeTZDtype):
            self = cast('DatetimeArray', self)
            naive = self.tz_localize(None)
            result = naive._round(freq, mode, ambiguous, nonexistent)
            return result.tz_localize(self.tz, ambiguous=ambiguous, nonexistent=nonexistent)
        values = self.view('i8')
        values = cast(np.ndarray, values)
        nanos = get_unit_for_round(freq, self._creso)
        if nanos == 0:
            return self.copy()
        result_i8 = round_nsint64(values, mode, nanos)
        result = self._maybe_mask_results(result_i8, fill_value=iNaT)
        result = result.view(self._ndarray.dtype)
        return self._simple_new(result, dtype=self.dtype)

    @Appender((_round_doc + _round_example).format(op='round'))
    def round(self, freq, ambiguous: TimeAmbiguous='raise', nonexistent: TimeNonexistent='raise') -> Self:
        return self._round(freq, RoundTo.NEAREST_HALF_EVEN, ambiguous, nonexistent)

    @Appender((_round_doc + _floor_example).format(op='floor'))
    def floor(self, freq, ambiguous: TimeAmbiguous='raise', nonexistent: TimeNonexistent='raise') -> Self:
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)

    @Appender((_round_doc + _ceil_example).format(op='ceil'))
    def ceil(self, freq, ambiguous: TimeAmbiguous='raise', nonexistent: TimeNonexistent='raise') -> Self:
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)

    def any(self, *, axis: AxisInt | None=None, skipna: bool=True) -> bool:
        return nanops.nanany(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    def all(self, *, axis: AxisInt | None=None, skipna: bool=True) -> bool:
        return nanops.nanall(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    def _maybe_clear_freq(self) -> None:
        self._freq = None

    def _with_freq(self, freq) -> Self:
        """
        Helper to get a view on the same data, with a new freq.

        Parameters
        ----------
        freq : DateOffset, None, or "infer"

        Returns
        -------
        Same type as self
        """
        if freq is None:
            pass
        elif len(self) == 0 and isinstance(freq, BaseOffset):
            if self.dtype.kind == 'm' and (not isinstance(freq, Tick)):
                raise TypeError('TimedeltaArray/Index freq must be a Tick')
        else:
            assert freq == 'infer'
            freq = to_offset(self.inferred_freq)
        arr = self.view()
        arr._freq = freq
        return arr

    def _values_for_json(self) -> np.ndarray:
        if isinstance(self.dtype, np.dtype):
            return self._ndarray
        return super()._values_for_json()

    def factorize(self, use_na_sentinel: bool=True, sort: bool=False):
        if self.freq is not None:
            codes = np.arange(len(self), dtype=np.intp)
            uniques = self.copy()
            if sort and self.freq.n < 0:
                codes = codes[::-1]
                uniques = uniques[::-1]
            return (codes, uniques)
        if sort:
            raise NotImplementedError(f"The 'sort' keyword in {type(self).__name__}.factorize is ignored unless arr.freq is not None. To factorize with sort, call pd.factorize(obj, sort=True) instead.")
        return super().factorize(use_na_sentinel=use_na_sentinel)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt=0) -> Self:
        new_obj = super()._concat_same_type(to_concat, axis)
        obj = to_concat[0]
        if axis == 0:
            to_concat = [x for x in to_concat if len(x)]
            if obj.freq is not None and all((x.freq == obj.freq for x in to_concat)):
                pairs = zip(to_concat[:-1], to_concat[1:])
                if all((pair[0][-1] + obj.freq == pair[1][0] for pair in pairs)):
                    new_freq = obj.freq
                    new_obj._freq = new_freq
        return new_obj

    def copy(self, order: str='C') -> Self:
        new_obj = super().copy(order=order)
        new_obj._freq = self.freq
        return new_obj

    def interpolate(self, *, method: InterpolateOptions, axis: int, index: Index, limit, limit_direction, limit_area, copy: bool, **kwargs) -> Self:
        """
        See NDFrame.interpolate.__doc__.
        """
        if method != 'linear':
            raise NotImplementedError
        if not copy:
            out_data = self._ndarray
        else:
            out_data = self._ndarray.copy()
        missing.interpolate_2d_inplace(out_data, method=method, axis=axis, index=index, limit=limit, limit_direction=limit_direction, limit_area=limit_area, **kwargs)
        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

    @property
    def _is_dates_only(self) -> bool:
        """
        Check if we are round times at midnight (and no timezone), which will
        be given a more compact __repr__ than other cases. For TimedeltaArray
        we are checking for multiples of 24H.
        """
        if not lib.is_np_dtype(self.dtype):
            return False
        values_int = self.asi8
        consider_values = values_int != iNaT
        reso = get_unit_from_dtype(self.dtype)
        ppd = periods_per_day(reso)
        even_days = np.logical_and(consider_values, values_int % ppd != 0).sum() == 0
        return even_days