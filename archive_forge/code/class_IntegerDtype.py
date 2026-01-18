from __future__ import annotations
from typing import ClassVar
import numpy as np
from pandas.core.dtypes.base import register_extension_dtype
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.arrays.numeric import (
class IntegerDtype(NumericDtype):
    """
    An ExtensionDtype to hold a single size & kind of integer dtype.

    These specific implementations are subclasses of the non-public
    IntegerDtype. For example, we have Int8Dtype to represent signed int 8s.

    The attributes name & type are set when these subclasses are created.
    """
    _default_np_dtype = np.dtype(np.int64)
    _checker = is_integer_dtype

    @classmethod
    def construct_array_type(cls) -> type[IntegerArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return IntegerArray

    @classmethod
    def _get_dtype_mapping(cls) -> dict[np.dtype, IntegerDtype]:
        return NUMPY_INT_TO_DTYPE

    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless. e.g. if 'values'
        has a floating dtype, each value must be an integer.
        """
        try:
            return values.astype(dtype, casting='safe', copy=copy)
        except TypeError as err:
            casted = values.astype(dtype, copy=copy)
            if (casted == values).all():
                return casted
            raise TypeError(f'cannot safely cast non-equivalent {values.dtype} to {np.dtype(dtype)}') from err