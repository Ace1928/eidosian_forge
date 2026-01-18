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
def _window_row_number(self, df: DataFrame, func: WindowFunctionSpec, args: List[ArgumentSpec], dest_col_name: str) -> DataFrame:
    assert_or_throw(len(args) == 0, ValueError(f'{args}'))
    assert_or_throw(not func.has_windowframe, ValueError("row_number can't have windowframe"))
    assert_or_throw(not (func.has_order_by and (not func.has_partition_by)), ValueError('for dask engine, order by requires partition by'))
    assert_or_throw(func.has_partition_by, ValueError('for dask engine, partition by is required for row_number()'))
    keys = list(df.keys())
    ndf = self.to_native(df)
    ndf, sort_keys, asc = self._prepare_sort(ndf, func.window.order_by)
    gp, gp_keys, _ = self._safe_groupby(ndf, func.window.partition_keys)
    if not func.has_order_by:
        col = gp.cumcount() + 1
        ndf[dest_col_name] = col
        return self.to_df(ndf[keys + [dest_col_name]])

    def gp_apply(df: Any) -> Any:
        df = df.sort_values(sort_keys, ascending=asc)
        col = np.arange(1, 1 + df.shape[0])
        df[dest_col_name] = col
        return df.reset_index(drop=True)
    meta = [(ndf[x].name, ndf[x].dtype) for x in ndf.columns]
    meta += [(dest_col_name, 'int64')]
    ndf = gp.apply(gp_apply, meta=meta).reset_index(drop=True)
    return self.to_df(ndf[keys + [dest_col_name]])