from __future__ import annotations
import datetime
import warnings
from numbers import Integral
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas.core.window import Rolling as pd_Rolling
from dask.array.core import normalize_arg
from dask.base import tokenize
from dask.blockwise import BlockwiseDepDict
from dask.dataframe import methods
from dask.dataframe._compat import check_axis_keyword_deprecation
from dask.dataframe.core import (
from dask.dataframe.io import from_pandas
from dask.dataframe.multi import _maybe_align_partitions
from dask.dataframe.utils import (
from dask.delayed import unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import M, apply, derived_from, funcname, has_keyword
def _handle_frame_argument(arg):
    dsk = {}
    prevs_parts_dsk, prevs = _get_previous_partitions(arg, before)
    dsk.update(prevs_parts_dsk)
    nexts_parts_dsk, nexts = _get_nexts_partitions(arg, after)
    dsk.update(nexts_parts_dsk)
    name_a = 'overlap-concat-' + tokenize(arg)
    for i, (prev, current, next) in enumerate(zip(prevs, arg.__dask_keys__(), nexts)):
        key = (name_a, i)
        dsk[key] = (_combined_parts, prev, current, next, before, after)
    graph = HighLevelGraph.from_collections(name_a, dsk, dependencies=[arg])
    return new_dd_object(graph, name_a, meta, divisions)