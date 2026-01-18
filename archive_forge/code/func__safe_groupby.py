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
def _safe_groupby(self, ndf: Any, keys: List[str]) -> Tuple[Any, List[str], pandas.MultiIndex]:
    nulldf = ndf[keys].isnull()
    gp_keys: List[str] = []
    orig_keys = list(ndf.columns)
    for k in keys:
        ndf[k + '_n'] = nulldf[k]
        ndf[k + '_g'] = ndf[k].fillna(0)
        gp_keys.append(k + '_n')
        gp_keys.append(k + '_g')
    ndf = ndf[orig_keys + gp_keys]
    mi_data = [numpy.ndarray(0, ndf[x].dtype) for x in gp_keys]
    mi = pandas.MultiIndex.from_arrays(mi_data, names=gp_keys)
    return (ndf.groupby(gp_keys), gp_keys, mi)