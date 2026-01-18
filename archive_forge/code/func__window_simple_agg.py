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
def _window_simple_agg(self, agg_func: Callable, df: DataFrame, func: WindowFunctionSpec, args: List[ArgumentSpec], dest_col_name: str) -> DataFrame:
    assert_or_throw(not (func.has_order_by and (not func.has_partition_by)), ValueError('for dask engine, order by requires partition by'))
    assert_or_throw(func.has_partition_by, ValueError('for dask engine, partition by is required for window aggregation'))
    assert_or_throw(len(args) == 1 and args[0].is_col, ValueError(f'{args}'))
    assert_or_throw(not func.unique and (not func.dropna), NotImplementedError(f"window function doesn't support unique {func.unique} " + f'and dropna {func.dropna}'))
    col = args[0].value
    ndf = self.to_native(df)
    keys = list(df.keys())
    ndf, sort_keys, asc = self._prepare_sort(ndf, func.window.order_by)
    gp, gp_keys, mi = self._safe_groupby(ndf, func.window.partition_keys)

    def gp_apply(gdf: Any) -> Any:
        if len(sort_keys) > 0:
            gdf = gdf.sort_values(sort_keys, ascending=asc)
        if not func.has_windowframe:
            res: Any = agg_func(gdf[col])
        else:
            wf = func.window.windowframe
            assert_or_throw(isinstance(wf, RollingWindowFrame), NotImplementedError(f'{wf}'))
            min_p = 0 if func.name in ['count', 'min', 'max'] else 1
            if wf.start is None and wf.end == 0:
                rolling_func: Callable = lambda x: x.expanding(min_periods=min_p)
            elif wf.window is not None and wf.end == 0:
                rolling_func: Callable = lambda x: x.rolling(window=wf.window, min_periods=min_p)
            else:
                rolling_func: Callable = lambda x: x.rolling(window=_RowsIndexer(wf.start, wf.end), min_periods=min_p)
            res = agg_func(rolling_func(gdf[col]))
        gdf[dest_col_name] = res
        return gdf.reset_index(drop=True)
    meta = [(ndf[x].name, ndf[x].dtype) for x in ndf.columns]
    meta += [(dest_col_name, ndf[col].dtype)]
    ndf = gp.apply(gp_apply, meta=meta).reset_index(drop=True)
    return self.to_df(ndf[keys + [dest_col_name]])