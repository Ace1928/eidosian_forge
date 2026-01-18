from typing import Any, Callable, Dict, List, Tuple
import dask.array as np
import dask.dataframe as pd
import numpy
import pandas
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd import QPDEngine, run_sql
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_pandas.engine import _RowsIndexer
def _agg_no_gp(self, func: AggFunctionSpec, x: Any) -> Any:
    name = func.name.lower()
    if name == 'sum':
        if not func.unique:
            return x.to_frame().sum(min_count=1).compute()[0]
        else:
            return x.to_frame().drop_duplicates().sum(min_count=1).compute()[0]
    if name in ['avg', 'mean']:
        if not func.unique:
            return x.mean()
        else:
            return x.drop_duplicates().mean()
    if name in ['first', 'first_value']:
        if func.dropna:
            return x.dropna().head(1, compute=False)
        else:
            return x.head(1, compute=False)
    if name in ['last', 'last_value']:
        if func.dropna:
            return x.dropna().tail(1, compute=False)
        else:
            return x.tail(1, compute=False)
    if name == 'count':
        if not func.unique and (not func.dropna):
            return x.size
        if func.unique and (not func.dropna):
            return x.drop_duplicates().size
        if func.unique and func.dropna:
            return x.nunique()
        if not func.unique and func.dropna:
            return x.count()
    if name == 'min':
        return x.dropna().min()
    if name == 'max':
        return x.dropna().max()
    raise NotImplementedError