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
@register_extension_dtype
class BooleanDtype(BaseMaskedDtype):
    """
    Extension dtype for boolean data.

    .. warning::

       BooleanDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.BooleanDtype()
    BooleanDtype
    """
    name: ClassVar[str] = 'boolean'

    @property
    def type(self) -> type:
        return np.bool_

    @property
    def kind(self) -> str:
        return 'b'

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype('bool')

    @classmethod
    def construct_array_type(cls) -> type_t[BooleanArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return BooleanArray

    def __repr__(self) -> str:
        return 'BooleanDtype'

    @property
    def _is_boolean(self) -> bool:
        return True

    @property
    def _is_numeric(self) -> bool:
        return True

    def __from_arrow__(self, array: pyarrow.Array | pyarrow.ChunkedArray) -> BooleanArray:
        """
        Construct BooleanArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow
        if array.type != pyarrow.bool_() and (not pyarrow.types.is_null(array.type)):
            raise TypeError(f'Expected array of boolean type, got {array.type} instead')
        if isinstance(array, pyarrow.Array):
            chunks = [array]
            length = len(array)
        else:
            chunks = array.chunks
            length = array.length()
        if pyarrow.types.is_null(array.type):
            mask = np.ones(length, dtype=bool)
            data = np.empty(length, dtype=bool)
            return BooleanArray(data, mask)
        results = []
        for arr in chunks:
            buflist = arr.buffers()
            data = pyarrow.BooleanArray.from_buffers(arr.type, len(arr), [None, buflist[1]], offset=arr.offset).to_numpy(zero_copy_only=False)
            if arr.null_count != 0:
                mask = pyarrow.BooleanArray.from_buffers(arr.type, len(arr), [None, buflist[0]], offset=arr.offset).to_numpy(zero_copy_only=False)
                mask = ~mask
            else:
                mask = np.zeros(len(arr), dtype=bool)
            bool_arr = BooleanArray(data, mask)
            results.append(bool_arr)
        if not results:
            return BooleanArray(np.array([], dtype=np.bool_), np.array([], dtype=np.bool_))
        else:
            return BooleanArray._concat_same_type(results)