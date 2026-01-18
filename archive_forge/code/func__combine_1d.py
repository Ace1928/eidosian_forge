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
def _combine_1d(datasets, concat_dim, compat: CompatOptions='no_conflicts', data_vars='all', coords='different', fill_value=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='drop'):
    """
    Applies either concat or merge to 1D list of datasets depending on value
    of concat_dim
    """
    if concat_dim is not None:
        try:
            combined = concat(datasets, dim=concat_dim, data_vars=data_vars, coords=coords, compat=compat, fill_value=fill_value, join=join, combine_attrs=combine_attrs)
        except ValueError as err:
            if 'encountered unexpected variable' in str(err):
                raise ValueError('These objects cannot be combined using only xarray.combine_nested, instead either use xarray.combine_by_coords, or do it manually with xarray.concat, xarray.merge and xarray.align')
            else:
                raise
    else:
        combined = merge(datasets, compat=compat, fill_value=fill_value, join=join, combine_attrs=combine_attrs)
    return combined