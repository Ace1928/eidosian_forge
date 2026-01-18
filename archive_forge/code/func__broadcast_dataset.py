from __future__ import annotations
import functools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, TypeVar, cast, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes
from xarray.core.indexes import (
from xarray.core.types import T_Alignable
from xarray.core.utils import is_dict_like, is_full_slice
from xarray.core.variable import Variable, as_compatible_data, calculate_dimensions
def _broadcast_dataset(ds: T_Dataset) -> T_Dataset:
    data_vars = {k: _set_dims(ds.variables[k]) for k in ds.data_vars}
    coords = dict(ds.coords)
    coords.update(common_coords)
    return ds.__class__(data_vars, coords, ds.attrs)