from __future__ import annotations
import numbers
from typing import (
import numpy as np
from pandas._libs import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import (
from pandas.core.arrays.masked import (
class NumericArray(BaseMaskedArray):
    """
    Base class for IntegerArray and FloatingArray.
    """
    _dtype_cls: type[NumericDtype]

    def __init__(self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool=False) -> None:
        checker = self._dtype_cls._checker
        if not (isinstance(values, np.ndarray) and checker(values.dtype)):
            descr = 'floating' if self._dtype_cls.kind == 'f' else 'integer'
            raise TypeError(f"values should be {descr} numpy array. Use the 'pd.array' function instead")
        if values.dtype == np.float16:
            raise TypeError('FloatingArray does not support np.float16 dtype.')
        super().__init__(values, mask, copy=copy)

    @cache_readonly
    def dtype(self) -> NumericDtype:
        mapping = self._dtype_cls._get_dtype_mapping()
        return mapping[self._data.dtype]

    @classmethod
    def _coerce_to_array(cls, value, *, dtype: DtypeObj, copy: bool=False) -> tuple[np.ndarray, np.ndarray]:
        dtype_cls = cls._dtype_cls
        default_dtype = dtype_cls._default_np_dtype
        values, mask, _, _ = _coerce_to_data_and_mask(value, dtype, copy, dtype_cls, default_dtype)
        return (values, mask)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None=None, copy: bool=False) -> Self:
        from pandas.core.tools.numeric import to_numeric
        scalars = to_numeric(strings, errors='raise', dtype_backend='numpy_nullable')
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)
    _HANDLED_TYPES = (np.ndarray, numbers.Number)