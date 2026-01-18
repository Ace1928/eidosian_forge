from __future__ import annotations
import functools
import itertools
import operator
import warnings
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, cast, overload
import numpy as np
from xarray.core import dtypes, duck_array_ops, utils
from xarray.core.alignment import align, deep_align
from xarray.core.common import zeros_like
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.formatting import limit_lines
from xarray.core.indexes import Index, filter_indexes_from_coords
from xarray.core.merge import merge_attrs, merge_coordinates_without_align
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import Dims, T_DataArray
from xarray.core.utils import is_dict_like, is_duck_dask_array, is_scalar, parse_dims
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.util.deprecation_helpers import deprecate_dims
def _cov_corr(da_a: T_DataArray, da_b: T_DataArray, weights: T_DataArray | None=None, dim: Dims=None, ddof: int=0, method: Literal['cov', 'corr', None]=None) -> T_DataArray:
    """
    Internal method for xr.cov() and xr.corr() so only have to
    sanitize the input arrays once and we don't repeat code.
    """
    da_a, da_b = align(da_a, da_b, join='inner', copy=False)
    valid_values = da_a.notnull() & da_b.notnull()
    da_a = da_a.where(valid_values)
    da_b = da_b.where(valid_values)
    if weights is not None:
        demeaned_da_a = da_a - da_a.weighted(weights).mean(dim=dim)
        demeaned_da_b = da_b - da_b.weighted(weights).mean(dim=dim)
    else:
        demeaned_da_a = da_a - da_a.mean(dim=dim)
        demeaned_da_b = da_b - da_b.mean(dim=dim)
    if weights is not None:
        cov = (demeaned_da_a.conj() * demeaned_da_b).weighted(weights).mean(dim=dim, skipna=True)
    else:
        cov = (demeaned_da_a.conj() * demeaned_da_b).mean(dim=dim, skipna=True)
    if method == 'cov':
        valid_count = valid_values.sum(dim)
        adjust = valid_count / (valid_count - ddof)
        return cast(T_DataArray, cov * adjust)
    else:
        if weights is not None:
            da_a_std = da_a.weighted(weights).std(dim=dim)
            da_b_std = da_b.weighted(weights).std(dim=dim)
        else:
            da_a_std = da_a.std(dim=dim)
            da_b_std = da_b.std(dim=dim)
        corr = cov / (da_a_std * da_b_std)
        return cast(T_DataArray, corr)