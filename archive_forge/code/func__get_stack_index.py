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
def _get_stack_index(self, dim, multi=False, create_index=False) -> tuple[Index | None, dict[Hashable, Variable]]:
    """Used by stack and unstack to get one pandas (multi-)index among
        the indexed coordinates along dimension `dim`.

        If exactly one index is found, return it with its corresponding
        coordinate variables(s), otherwise return None and an empty dict.

        If `create_index=True`, create a new index if none is found or raise
        an error if multiple indexes are found.

        """
    stack_index: Index | None = None
    stack_coords: dict[Hashable, Variable] = {}
    for name, index in self._indexes.items():
        var = self._variables[name]
        if var.ndim == 1 and var.dims[0] == dim and (not multi and (not self.xindexes.is_multi(name)) or (multi and type(index).unstack is not Index.unstack)):
            if stack_index is not None and index is not stack_index:
                if create_index:
                    raise ValueError(f'cannot stack dimension {dim!r} with `create_index=True` and with more than one index found along that dimension')
                return (None, {})
            stack_index = index
            stack_coords[name] = var
    if create_index and stack_index is None:
        if dim in self._variables:
            var = self._variables[dim]
        else:
            _, _, var = _get_virtual_variable(self._variables, dim, self.sizes)
        stack_index = PandasIndex([0], dim)
        stack_coords = {dim: var}
    return (stack_index, stack_coords)