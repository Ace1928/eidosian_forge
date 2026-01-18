from __future__ import annotations
from collections.abc import (
from typing import (
import numpy as np
from pandas._libs import algos as libalgos
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import BaseMaskedDtype
class SelectNSeries(SelectN):
    """
    Implement n largest/smallest for Series

    Parameters
    ----------
    obj : Series
    n : int
    keep : {'first', 'last'}, default 'first'

    Returns
    -------
    nordered : Series
    """

    def compute(self, method: str) -> Series:
        from pandas.core.reshape.concat import concat
        n = self.n
        dtype = self.obj.dtype
        if not self.is_valid_dtype_n_method(dtype):
            raise TypeError(f"Cannot use method '{method}' with dtype {dtype}")
        if n <= 0:
            return self.obj[[]]
        dropped = self.obj.dropna()
        nan_index = self.obj.drop(dropped.index)
        if n >= len(self.obj):
            ascending = method == 'nsmallest'
            return self.obj.sort_values(ascending=ascending).head(n)
        new_dtype = dropped.dtype
        arr = dropped._values
        if needs_i8_conversion(arr.dtype):
            arr = arr.view('i8')
        elif isinstance(arr.dtype, BaseMaskedDtype):
            arr = arr._data
        else:
            arr = np.asarray(arr)
        if arr.dtype.kind == 'b':
            arr = arr.view(np.uint8)
        if method == 'nlargest':
            arr = -arr
            if is_integer_dtype(new_dtype):
                arr -= 1
            elif is_bool_dtype(new_dtype):
                arr = 1 - -arr
        if self.keep == 'last':
            arr = arr[::-1]
        nbase = n
        narr = len(arr)
        n = min(n, narr)
        if len(arr) > 0:
            kth_val = libalgos.kth_smallest(arr.copy(order='C'), n - 1)
        else:
            kth_val = np.nan
        ns, = np.nonzero(arr <= kth_val)
        inds = ns[arr[ns].argsort(kind='mergesort')]
        if self.keep != 'all':
            inds = inds[:n]
            findex = nbase
        elif len(inds) < nbase <= len(nan_index) + len(inds):
            findex = len(nan_index) + len(inds)
        else:
            findex = len(inds)
        if self.keep == 'last':
            inds = narr - 1 - inds
        return concat([dropped.iloc[inds], nan_index]).iloc[:findex]