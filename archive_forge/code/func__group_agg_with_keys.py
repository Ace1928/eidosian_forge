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
def _group_agg_with_keys(self, df: DataFrame, keys: List[str], agg_map: Dict[str, Tuple[str, AggFunctionSpec]]) -> DataFrame:
    ndf = self.to_native(df)
    gp, gp_keys, mi = self._safe_groupby(ndf, keys)
    col_group1: List[pd.Series] = []
    col_group2: List[pd.Series] = []
    for k, v in agg_map.items():
        if '*' in v[0] or ',' in v[0]:
            continue
        series, is_agg = self._agg_gp(v[1], gp[v[0]], ndf[v[0]].dtype, mi)
        series = series.rename(k)
        if is_agg:
            col_group1.append(series)
        else:
            col_group2.append(series)
    for k, v in agg_map.items():
        if '*' in v[0] or ',' in v[0]:
            if v[1].unique:
                series = self._count_unique(ndf, gp_keys, v[0], v[1]).rename(k)
                col_group2.append(series)
            else:
                series = self._count_all(gp, v[1]).rename(k)
                col_group1.append(series)
    res: Any = None
    if len(col_group1) > 0:
        res = col_group1[0].to_frame()
        for s in col_group1[1:]:
            res[s.name] = s
        res = res.reset_index()
    if len(col_group2) > 0:
        if res is None:
            res = col_group2[0].reset_index()
            n = 1
        else:
            n = 0
        for s in col_group2[n:]:
            tdf = s.reset_index()
            res = res.merge(tdf, 'outer', on=gp_keys)
    assert_or_throw(res is not None, ValueError('No aggregation happened'))
    res = res.reset_index()
    res = res[list(agg_map.keys())]
    return self.to_df(res)