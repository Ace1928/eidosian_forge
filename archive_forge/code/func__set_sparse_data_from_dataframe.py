from __future__ import annotations
import copy
import datetime
import inspect
import itertools
import math
import sys
import warnings
from collections import defaultdict
from collections.abc import (
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import IO, TYPE_CHECKING, Any, Callable, Generic, Literal, cast, overload
import numpy as np
import pandas as pd
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
from xarray.core import (
from xarray.core import dtypes as xrdtypes
from xarray.core._aggregations import DatasetAggregations
from xarray.core.alignment import (
from xarray.core.arithmetic import DatasetArithmetic
from xarray.core.common import (
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import (
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.indexes import (
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import (
from xarray.core.missing import get_clean_interp_index
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import (
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import array_type, is_chunked_array
from xarray.plot.accessor import DatasetPlotAccessor
from xarray.util.deprecation_helpers import _deprecate_positional_args
def _set_sparse_data_from_dataframe(self, idx: pd.Index, arrays: list[tuple[Hashable, np.ndarray]], dims: tuple) -> None:
    from sparse import COO
    if isinstance(idx, pd.MultiIndex):
        coords = np.stack([np.asarray(code) for code in idx.codes], axis=0)
        is_sorted = idx.is_monotonic_increasing
        shape = tuple((lev.size for lev in idx.levels))
    else:
        coords = np.arange(idx.size).reshape(1, -1)
        is_sorted = True
        shape = (idx.size,)
    for name, values in arrays:
        dtype, fill_value = xrdtypes.maybe_promote(values.dtype)
        values = np.asarray(values, dtype=dtype)
        data = COO(coords, values, shape, has_duplicates=False, sorted=is_sorted, fill_value=fill_value)
        self[name] = (dims, data)