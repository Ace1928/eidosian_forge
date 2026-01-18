from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasUtils
from qpd import QPDEngine, run_sql
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
def _safe_sort(self, ndf: Any, order_by: OrderBySpec) -> Any:
    if len(order_by.keys) == 0:
        return ndf
    keys = list(ndf.columns)
    okeys: List[str] = []
    asc: List[bool] = []
    for oi in order_by:
        nk = oi.name + '_null'
        ndf[nk] = ndf[oi.name].isnull()
        okeys.append(nk)
        asc.append(oi.pd_na_position != 'first')
        okeys.append(oi.name)
        asc.append(oi.asc)
    ndf = ndf.sort_values(okeys, ascending=asc)
    return ndf[keys]