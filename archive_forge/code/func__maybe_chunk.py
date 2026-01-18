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
def _maybe_chunk(name, var, chunks, token=None, lock=None, name_prefix='xarray-', overwrite_encoded_chunks=False, inline_array=False, chunked_array_type: str | ChunkManagerEntrypoint | None=None, from_array_kwargs=None):
    from xarray.namedarray.daskmanager import DaskManager
    if chunks is not None:
        chunks = {dim: chunks[dim] for dim in var.dims if dim in chunks}
    if var.ndim:
        chunked_array_type = guess_chunkmanager(chunked_array_type)
        if isinstance(chunked_array_type, DaskManager):
            from dask.base import tokenize
            token2 = tokenize(token if token else var._data, str(chunks))
            name2 = f'{name_prefix}{name}-{token2}'
            from_array_kwargs = utils.consolidate_dask_from_array_kwargs(from_array_kwargs, name=name2, lock=lock, inline_array=inline_array)
        var = var.chunk(chunks, chunked_array_type=chunked_array_type, from_array_kwargs=from_array_kwargs)
        if overwrite_encoded_chunks and var.chunks is not None:
            var.encoding['chunks'] = tuple((x[0] for x in var.chunks))
        return var
    else:
        return var