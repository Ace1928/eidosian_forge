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
def _get_previous_partitions(df, before):
    """
    Helper to get the previous partitions required for the overlap
    """
    dsk = {}
    df_name = df._name
    name_a = 'overlap-prepend-' + tokenize(df, before)
    if before and isinstance(before, Integral):
        prevs = [None]
        for i in range(df.npartitions - 1):
            key = (name_a, i)
            dsk[key] = (M.tail, (df_name, i), before)
            prevs.append(key)
    elif isinstance(before, datetime.timedelta):
        divs = pd.Series(df.divisions)
        deltas = divs.diff().iloc[1:-1]
        if (before > deltas).any():
            pt_z = divs[0]
            prevs = [None]
            for i in range(df.npartitions - 1):
                pt_i = divs[i + 1]
                lb = max(pt_i - before, pt_z)
                first, j = (divs[i], i)
                while first > lb and j > 0:
                    first = first - deltas[j]
                    j = j - 1
                key = (name_a, i)
                dsk[key] = (_tail_timedelta, [(df_name, k) for k in range(j, i + 1)], (df_name, i + 1), before)
                prevs.append(key)
        else:
            prevs = [None]
            for i in range(df.npartitions - 1):
                key = (name_a, i)
                dsk[key] = (_tail_timedelta, [(df_name, i)], (df_name, i + 1), before)
                prevs.append(key)
    else:
        prevs = [None] * df.npartitions
    return (dsk, prevs)