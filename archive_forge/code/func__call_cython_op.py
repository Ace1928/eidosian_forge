from __future__ import annotations
import collections
import functools
from typing import (
import numpy as np
from pandas._libs import (
import pandas._libs.groupby as libgroupby
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import (
@final
def _call_cython_op(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask: npt.NDArray[np.bool_] | None, result_mask: npt.NDArray[np.bool_] | None, **kwargs) -> np.ndarray:
    orig_values = values
    dtype = values.dtype
    is_numeric = dtype.kind in 'iufcb'
    is_datetimelike = dtype.kind in 'mM'
    if is_datetimelike:
        values = values.view('int64')
        is_numeric = True
    elif dtype.kind == 'b':
        values = values.view('uint8')
    if values.dtype == 'float16':
        values = values.astype(np.float32)
    if self.how in ['any', 'all']:
        if mask is None:
            mask = isna(values)
        if dtype == object:
            if kwargs['skipna']:
                if mask.any():
                    values = values.copy()
                    values[mask] = True
        values = values.astype(bool, copy=False).view(np.int8)
        is_numeric = True
    values = values.T
    if mask is not None:
        mask = mask.T
        if result_mask is not None:
            result_mask = result_mask.T
    out_shape = self._get_output_shape(ngroups, values)
    func = self._get_cython_function(self.kind, self.how, values.dtype, is_numeric)
    values = self._get_cython_vals(values)
    out_dtype = self._get_out_dtype(values.dtype)
    result = maybe_fill(np.empty(out_shape, dtype=out_dtype))
    if self.kind == 'aggregate':
        counts = np.zeros(ngroups, dtype=np.int64)
        if self.how in ['idxmin', 'idxmax', 'min', 'max', 'mean', 'last', 'first', 'sum']:
            func(out=result, counts=counts, values=values, labels=comp_ids, min_count=min_count, mask=mask, result_mask=result_mask, is_datetimelike=is_datetimelike, **kwargs)
        elif self.how in ['sem', 'std', 'var', 'ohlc', 'prod', 'median']:
            if self.how in ['std', 'sem']:
                kwargs['is_datetimelike'] = is_datetimelike
            func(result, counts, values, comp_ids, min_count=min_count, mask=mask, result_mask=result_mask, **kwargs)
        elif self.how in ['any', 'all']:
            func(out=result, values=values, labels=comp_ids, mask=mask, result_mask=result_mask, **kwargs)
            result = result.astype(bool, copy=False)
        elif self.how in ['skew']:
            func(out=result, counts=counts, values=values, labels=comp_ids, mask=mask, result_mask=result_mask, **kwargs)
            if dtype == object:
                result = result.astype(object)
        else:
            raise NotImplementedError(f'{self.how} is not implemented')
    else:
        if self.how != 'rank':
            kwargs['result_mask'] = result_mask
        func(out=result, values=values, labels=comp_ids, ngroups=ngroups, is_datetimelike=is_datetimelike, mask=mask, **kwargs)
    if self.kind == 'aggregate' and self.how not in ['idxmin', 'idxmax']:
        if result.dtype.kind in 'iu' and (not is_datetimelike):
            cutoff = max(0 if self.how in ['sum', 'prod'] else 1, min_count)
            empty_groups = counts < cutoff
            if empty_groups.any():
                if result_mask is not None:
                    assert result_mask[empty_groups].all()
                else:
                    result = result.astype('float64')
                    result[empty_groups] = np.nan
    result = result.T
    if self.how not in self.cast_blocklist:
        res_dtype = self._get_result_dtype(orig_values.dtype)
        op_result = maybe_downcast_to_dtype(result, res_dtype)
    else:
        op_result = result
    return op_result