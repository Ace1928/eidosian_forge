from __future__ import annotations
import numbers
from typing import (
import numpy as np
from pandas._libs import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import (
from pandas.core.arrays.masked import (
class NumericDtype(BaseMaskedDtype):
    _default_np_dtype: np.dtype
    _checker: Callable[[Any], bool]

    def __repr__(self) -> str:
        return f'{self.name}Dtype()'

    @cache_readonly
    def is_signed_integer(self) -> bool:
        return self.kind == 'i'

    @cache_readonly
    def is_unsigned_integer(self) -> bool:
        return self.kind == 'u'

    @property
    def _is_numeric(self) -> bool:
        return True

    def __from_arrow__(self, array: pyarrow.Array | pyarrow.ChunkedArray) -> BaseMaskedArray:
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow
        from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
        array_class = self.construct_array_type()
        pyarrow_type = pyarrow.from_numpy_dtype(self.type)
        if not array.type.equals(pyarrow_type) and (not pyarrow.types.is_null(array.type)):
            rt_dtype = pandas_dtype(array.type.to_pandas_dtype())
            if rt_dtype.kind not in 'iuf':
                raise TypeError(f'Expected array of {self} type, got {array.type} instead')
            array = array.cast(pyarrow_type)
        if isinstance(array, pyarrow.ChunkedArray):
            if array.num_chunks == 0:
                array = pyarrow.array([], type=array.type)
            else:
                array = array.combine_chunks()
        data, mask = pyarrow_array_to_numpy_and_mask(array, dtype=self.numpy_dtype)
        return array_class(data.copy(), ~mask, copy=False)

    @classmethod
    def _get_dtype_mapping(cls) -> Mapping[np.dtype, NumericDtype]:
        raise AbstractMethodError(cls)

    @classmethod
    def _standardize_dtype(cls, dtype: NumericDtype | str | np.dtype) -> NumericDtype:
        """
        Convert a string representation or a numpy dtype to NumericDtype.
        """
        if isinstance(dtype, str) and dtype.startswith(('Int', 'UInt', 'Float')):
            dtype = dtype.lower()
        if not isinstance(dtype, NumericDtype):
            mapping = cls._get_dtype_mapping()
            try:
                dtype = mapping[np.dtype(dtype)]
            except KeyError as err:
                raise ValueError(f'invalid dtype specified {dtype}') from err
        return dtype

    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        """
        raise AbstractMethodError(cls)