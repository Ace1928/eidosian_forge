from __future__ import annotations
import itertools
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.concat import concat
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.merge import merge
from xarray.core.utils import iterate_nested
def _combine_single_variable_hypercube(datasets, fill_value=dtypes.NA, data_vars='all', coords='different', compat: CompatOptions='no_conflicts', join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='no_conflicts'):
    """
    Attempt to combine a list of Datasets into a hypercube using their
    coordinates.

    All provided Datasets must belong to a single variable, ie. must be
    assigned the same variable name. This precondition is not checked by this
    function, so the caller is assumed to know what it's doing.

    This function is NOT part of the public API.
    """
    if len(datasets) == 0:
        raise ValueError('At least one Dataset is required to resolve variable names for combined hypercube.')
    combined_ids, concat_dims = _infer_concat_order_from_coords(list(datasets))
    if fill_value is None:
        _check_shape_tile_ids(combined_ids)
    else:
        _check_dimension_depth_tile_ids(combined_ids)
    concatenated = _combine_nd(combined_ids, concat_dims=concat_dims, data_vars=data_vars, coords=coords, compat=compat, fill_value=fill_value, join=join, combine_attrs=combine_attrs)
    for dim in concat_dims:
        indexes = concatenated.indexes.get(dim)
        if not (indexes.is_monotonic_increasing or indexes.is_monotonic_decreasing):
            raise ValueError(f'Resulting object does not have monotonic global indexes along dimension {dim}')
    return concatenated