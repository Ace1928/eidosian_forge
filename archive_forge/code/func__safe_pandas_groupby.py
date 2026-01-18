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
def _safe_pandas_groupby(self, ndf: pandas.DataFrame, keys: List[str], keep_extra_keys: bool=False, as_index: bool=True, sort: bool=True) -> Tuple[Any, List[str]]:
    if not keep_extra_keys:
        ndf = ndf[ndf.columns]
    nulldf = ndf[keys].isnull()
    gp_keys: List[str] = []
    for k in keys:
        ndf[k + '_n'] = nulldf[k]
        ndf[k + '_g'] = ndf[k].fillna(0)
        gp_keys.append(k + '_n')
        gp_keys.append(k + '_g')
    return (ndf.groupby(gp_keys, as_index=as_index, sort=sort), gp_keys)