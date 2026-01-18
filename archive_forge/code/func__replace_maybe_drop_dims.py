from __future__ import annotations
import datetime
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from os import PathLike
from typing import (
import numpy as np
import pandas as pd
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import alignment, computation, dtypes, indexing, ops, utils
from xarray.core._aggregations import DataArrayAggregations
from xarray.core.accessor_dt import CombinedDatetimelikeAccessor
from xarray.core.accessor_str import StringAccessor
from xarray.core.alignment import (
from xarray.core.arithmetic import DataArrayArithmetic
from xarray.core.common import AbstractArray, DataWithCoords, get_chunksizes
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import (
from xarray.core.dataset import Dataset
from xarray.core.formatting import format_item
from xarray.core.indexes import (
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import PANDAS_TYPES, MergeError
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import (
from xarray.plot.accessor import DataArrayPlotAccessor
from xarray.plot.utils import _get_units_from_attrs
from xarray.util.deprecation_helpers import _deprecate_positional_args, deprecate_dims
def _replace_maybe_drop_dims(self, variable: Variable, name: Hashable | None | Default=_default) -> Self:
    if variable.dims == self.dims and variable.shape == self.shape:
        coords = self._coords.copy()
        indexes = self._indexes
    elif variable.dims == self.dims:
        new_sizes = dict(zip(self.dims, variable.shape))
        coords = {k: v for k, v in self._coords.items() if v.shape == tuple((new_sizes[d] for d in v.dims))}
        indexes = filter_indexes_from_coords(self._indexes, set(coords))
    else:
        allowed_dims = set(variable.dims)
        coords = {k: v for k, v in self._coords.items() if set(v.dims) <= allowed_dims}
        indexes = filter_indexes_from_coords(self._indexes, set(coords))
    return self._replace(variable, coords, name, indexes=indexes)