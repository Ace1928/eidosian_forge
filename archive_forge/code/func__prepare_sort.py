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
def _prepare_sort(self, ndf: Any, order_by: OrderBySpec) -> Tuple[Any, List[str], List[bool]]:
    if len(order_by.keys) == 0:
        return (ndf, [], [])
    okeys: List[str] = []
    asc: List[bool] = []
    for oi in order_by:
        nk = oi.name + '_null'
        ndf[nk] = ndf[oi.name].isnull()
        okeys.append(nk)
        asc.append(oi.pd_na_position != 'first')
        okeys.append(oi.name)
        asc.append(oi.asc)
    return (ndf, okeys, asc)