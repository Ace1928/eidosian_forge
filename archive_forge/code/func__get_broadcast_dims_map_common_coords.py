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
def _get_broadcast_dims_map_common_coords(args, exclude):
    common_coords = {}
    dims_map = {}
    for arg in args:
        for dim in arg.dims:
            if dim not in common_coords and dim not in exclude:
                dims_map[dim] = arg.sizes[dim]
                if dim in arg._indexes:
                    common_coords.update(arg.xindexes.get_all_coords(dim))
    return (dims_map, common_coords)