from __future__ import annotations
import copy
import datetime
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core import dtypes, duck_array_ops, nputils, ops
from xarray.core._aggregations import (
from xarray.core.alignment import align
from xarray.core.arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.concat import concat
from xarray.core.formatting import format_array_flat
from xarray.core.indexes import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import IndexVariable, Variable
from xarray.util.deprecation_helpers import _deprecate_positional_args
def _flox_reduce(self, dim: Dims, keep_attrs: bool | None=None, **kwargs: Any):
    """Adaptor function that translates our groupby API to that of flox."""
    import flox
    from flox.xarray import xarray_reduce
    from xarray.core.dataset import Dataset
    obj = self._original_obj
    grouper, = self.groupers
    isbin = isinstance(grouper.grouper, BinGrouper)
    if keep_attrs is None:
        keep_attrs = _get_keep_attrs(default=True)
    if Version(flox.__version__) < Version('0.9'):
        kwargs.setdefault('method', 'cohorts')
    numeric_only = kwargs.pop('numeric_only', None)
    if numeric_only:
        non_numeric = {name: var for name, var in obj.data_vars.items() if not (np.issubdtype(var.dtype, np.number) or var.dtype == np.bool_)}
    else:
        non_numeric = {}
    if 'min_count' in kwargs:
        if kwargs['func'] not in ['sum', 'prod']:
            raise TypeError("Received an unexpected keyword argument 'min_count'")
        elif kwargs['min_count'] is None:
            kwargs['min_count'] = 0
    if (dim is None or dim == grouper.name) and grouper.name in obj.xindexes:
        index = obj.indexes[grouper.name]
        if index.is_unique and self._squeeze:
            raise ValueError(f'cannot reduce over dimensions {grouper.name!r}')
    unindexed_dims: tuple[Hashable, ...] = tuple()
    if isinstance(grouper.group, _DummyGroup) and (not isbin):
        unindexed_dims = (grouper.name,)
    parsed_dim: tuple[Hashable, ...]
    if isinstance(dim, str):
        parsed_dim = (dim,)
    elif dim is None:
        parsed_dim = grouper.group.dims
    elif dim is ...:
        parsed_dim = tuple(obj.dims)
    else:
        parsed_dim = tuple(dim)
    if any((d not in grouper.group.dims and d not in obj.dims for d in parsed_dim)):
        raise ValueError(f'cannot reduce over dimensions {dim}.')
    if kwargs['func'] not in ['all', 'any', 'count']:
        kwargs.setdefault('fill_value', np.nan)
    if isbin and kwargs['func'] == 'count':
        kwargs.setdefault('fill_value', np.nan)
        kwargs.setdefault('min_count', 1)
    output_index = grouper.full_index
    result = xarray_reduce(obj.drop_vars(non_numeric.keys()), self._codes, dim=parsed_dim, expected_groups=(pd.RangeIndex(len(output_index)),), isbin=False, keep_attrs=keep_attrs, **kwargs)
    group_dims = grouper.group.dims
    if set(group_dims).issubset(set(parsed_dim)):
        result[grouper.name] = output_index
        result = result.drop_vars(unindexed_dims)
    for name, var in non_numeric.items():
        if all((d not in var.dims for d in parsed_dim)):
            result[name] = var.variable.set_dims((grouper.name,) + var.dims, (result.sizes[grouper.name],) + var.shape)
    if not isinstance(result, Dataset):
        result = self._restore_dim_order(result)
    return result