from __future__ import annotations
import numbers
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import register_extension_dtype
from pandas.core.dtypes.missing import isna
from pandas.core import ops
from pandas.core.array_algos import masked_accumulations
from pandas.core.arrays.masked import (
class BooleanArray(BaseMaskedArray):
    """
    Array of boolean (True/False) data with missing values.

    This is a pandas Extension array for boolean data, under the hood
    represented by 2 numpy arrays: a boolean array with the data and
    a boolean array with the mask (True indicating missing).

    BooleanArray implements Kleene logic (sometimes called three-value
    logic) for logical operations. See :ref:`boolean.kleene` for more.

    To construct an BooleanArray from generic array-like input, use
    :func:`pandas.array` specifying ``dtype="boolean"`` (see examples
    below).

    .. warning::

       BooleanArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d boolean-dtype array with the data.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values (True
        indicates missing).
    copy : bool, default False
        Whether to copy the `values` and `mask` arrays.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    BooleanArray

    Examples
    --------
    Create an BooleanArray with :func:`pandas.array`:

    >>> pd.array([True, False, None], dtype="boolean")
    <BooleanArray>
    [True, False, <NA>]
    Length: 3, dtype: boolean
    """
    _internal_fill_value = False
    _truthy_value = True
    _falsey_value = False
    _TRUE_VALUES = {'True', 'TRUE', 'true', '1', '1.0'}
    _FALSE_VALUES = {'False', 'FALSE', 'false', '0', '0.0'}

    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self:
        result = super()._simple_new(values, mask)
        result._dtype = BooleanDtype()
        return result

    def __init__(self, values: np.ndarray, mask: np.ndarray, copy: bool=False) -> None:
        if not (isinstance(values, np.ndarray) and values.dtype == np.bool_):
            raise TypeError("values should be boolean numpy array. Use the 'pd.array' function instead")
        self._dtype = BooleanDtype()
        super().__init__(values, mask, copy=copy)

    @property
    def dtype(self) -> BooleanDtype:
        return self._dtype

    @classmethod
    def _from_sequence_of_strings(cls, strings: list[str], *, dtype: Dtype | None=None, copy: bool=False, true_values: list[str] | None=None, false_values: list[str] | None=None) -> BooleanArray:
        true_values_union = cls._TRUE_VALUES.union(true_values or [])
        false_values_union = cls._FALSE_VALUES.union(false_values or [])

        def map_string(s) -> bool:
            if s in true_values_union:
                return True
            elif s in false_values_union:
                return False
            else:
                raise ValueError(f'{s} cannot be cast to bool')
        scalars = np.array(strings, dtype=object)
        mask = isna(scalars)
        scalars[~mask] = list(map(map_string, scalars[~mask]))
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)
    _HANDLED_TYPES = (np.ndarray, numbers.Number, bool, np.bool_)

    @classmethod
    def _coerce_to_array(cls, value, *, dtype: DtypeObj, copy: bool=False) -> tuple[np.ndarray, np.ndarray]:
        if dtype:
            assert dtype == 'boolean'
        return coerce_to_array(value, copy=copy)

    def _logical_method(self, other, op):
        assert op.__name__ in {'or_', 'ror_', 'and_', 'rand_', 'xor', 'rxor'}
        other_is_scalar = lib.is_scalar(other)
        mask = None
        if isinstance(other, BooleanArray):
            other, mask = (other._data, other._mask)
        elif is_list_like(other):
            other = np.asarray(other, dtype='bool')
            if other.ndim > 1:
                raise NotImplementedError('can only perform ops with 1-d structures')
            other, mask = coerce_to_array(other, copy=False)
        elif isinstance(other, np.bool_):
            other = other.item()
        if other_is_scalar and other is not libmissing.NA and (not lib.is_bool(other)):
            raise TypeError(f"'other' should be pandas.NA or a bool. Got {type(other).__name__} instead.")
        if not other_is_scalar and len(self) != len(other):
            raise ValueError('Lengths must match')
        if op.__name__ in {'or_', 'ror_'}:
            result, mask = ops.kleene_or(self._data, other, self._mask, mask)
        elif op.__name__ in {'and_', 'rand_'}:
            result, mask = ops.kleene_and(self._data, other, self._mask, mask)
        else:
            result, mask = ops.kleene_xor(self._data, other, self._mask, mask)
        return self._maybe_mask_result(result, mask)

    def _accumulate(self, name: str, *, skipna: bool=True, **kwargs) -> BaseMaskedArray:
        data = self._data
        mask = self._mask
        if name in ('cummin', 'cummax'):
            op = getattr(masked_accumulations, name)
            data, mask = op(data, mask, skipna=skipna, **kwargs)
            return self._simple_new(data, mask)
        else:
            from pandas.core.arrays import IntegerArray
            return IntegerArray(data.astype(int), mask)._accumulate(name, skipna=skipna, **kwargs)