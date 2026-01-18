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
def _validate_interp_indexers(self, indexers: Mapping[Any, Any]) -> Iterator[tuple[Hashable, Variable]]:
    """Variant of _validate_indexers to be used for interpolation"""
    for k, v in self._validate_indexers(indexers):
        if isinstance(v, Variable):
            if v.ndim == 1:
                yield (k, v.to_index_variable())
            else:
                yield (k, v)
        elif isinstance(v, int):
            yield (k, Variable((), v, attrs=self.coords[k].attrs))
        elif isinstance(v, np.ndarray):
            if v.ndim == 0:
                yield (k, Variable((), v, attrs=self.coords[k].attrs))
            elif v.ndim == 1:
                yield (k, IndexVariable((k,), v, attrs=self.coords[k].attrs))
            else:
                raise AssertionError()
        else:
            raise TypeError(type(v))