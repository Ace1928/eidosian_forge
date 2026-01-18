from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.core.resample import Resampler as pd_Resampler
from dask.base import tokenize
from dask.dataframe import methods
from dask.dataframe._compat import PANDAS_GE_140
from dask.dataframe.core import DataFrame, Series
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from
def _resample_series(series, start, end, reindex_closed, rule, resample_kwargs, how, fill_value, how_args, how_kwargs):
    out = getattr(series.resample(rule, **resample_kwargs), how)(*how_args, **how_kwargs)
    if PANDAS_GE_140:
        if reindex_closed is None:
            inclusive = 'both'
        else:
            inclusive = reindex_closed
        closed_kwargs = {'inclusive': inclusive}
    else:
        closed_kwargs = {'closed': reindex_closed}
    new_index = pd.date_range(start.tz_localize(None), end.tz_localize(None), freq=rule, **closed_kwargs, name=out.index.name).tz_localize(start.tz, nonexistent='shift_forward')
    if not out.index.isin(new_index).all():
        raise ValueError("Index is not contained within new index. This can often be resolved by using larger partitions, or unambiguous frequencies: 'Q', 'A'...")
    return out.reindex(new_index, fill_value=fill_value)