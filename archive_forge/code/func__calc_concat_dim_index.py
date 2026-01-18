from __future__ import annotations
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, utils
from xarray.core.alignment import align, reindex_variables
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.core.indexes import Index, PandasIndex
from xarray.core.merge import (
from xarray.core.types import T_DataArray, T_Dataset, T_Variable
from xarray.core.variable import Variable
from xarray.core.variable import concat as concat_vars
def _calc_concat_dim_index(dim_or_data: Hashable | Any) -> tuple[Hashable, PandasIndex | None]:
    """Infer the dimension name and 1d index / coordinate variable (if appropriate)
    for concatenating along the new dimension.

    """
    from xarray.core.dataarray import DataArray
    dim: Hashable | None
    if isinstance(dim_or_data, str):
        dim = dim_or_data
        index = None
    else:
        if not isinstance(dim_or_data, (DataArray, Variable)):
            dim = getattr(dim_or_data, 'name', None)
            if dim is None:
                dim = 'concat_dim'
        else:
            dim, = dim_or_data.dims
        coord_dtype = getattr(dim_or_data, 'dtype', None)
        index = PandasIndex(dim_or_data, dim, coord_dtype=coord_dtype)
    return (dim, index)