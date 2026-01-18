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
def _combine_nd(combined_ids, concat_dims, data_vars='all', coords='different', compat: CompatOptions='no_conflicts', fill_value=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='drop'):
    """
    Combines an N-dimensional structure of datasets into one by applying a
    series of either concat and merge operations along each dimension.

    No checks are performed on the consistency of the datasets, concat_dims or
    tile_IDs, because it is assumed that this has already been done.

    Parameters
    ----------
    combined_ids : Dict[Tuple[int, ...]], xarray.Dataset]
        Structure containing all datasets to be concatenated with "tile_IDs" as
        keys, which specify position within the desired final combined result.
    concat_dims : sequence of str
        The dimensions along which the datasets should be concatenated. Must be
        in order, and the length must match the length of the tuples used as
        keys in combined_ids. If the string is a dimension name then concat
        along that dimension, if it is None then merge.

    Returns
    -------
    combined_ds : xarray.Dataset
    """
    example_tile_id = next(iter(combined_ids.keys()))
    n_dims = len(example_tile_id)
    if len(concat_dims) != n_dims:
        raise ValueError(f'concat_dims has length {len(concat_dims)} but the datasets passed are nested in a {n_dims}-dimensional structure')
    for concat_dim in concat_dims:
        combined_ids = _combine_all_along_first_dim(combined_ids, dim=concat_dim, data_vars=data_vars, coords=coords, compat=compat, fill_value=fill_value, join=join, combine_attrs=combine_attrs)
    combined_ds, = combined_ids.values()
    return combined_ds