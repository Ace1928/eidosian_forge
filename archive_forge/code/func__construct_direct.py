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
@classmethod
def _construct_direct(cls, variables: dict[Any, Variable], coord_names: set[Hashable], dims: dict[Any, int] | None=None, attrs: dict | None=None, indexes: dict[Any, Index] | None=None, encoding: dict | None=None, close: Callable[[], None] | None=None) -> Self:
    """Shortcut around __init__ for internal use when we want to skip
        costly validation
        """
    if dims is None:
        dims = calculate_dimensions(variables)
    if indexes is None:
        indexes = {}
    obj = object.__new__(cls)
    obj._variables = variables
    obj._coord_names = coord_names
    obj._dims = dims
    obj._indexes = indexes
    obj._attrs = attrs
    obj._close = close
    obj._encoding = encoding
    return obj