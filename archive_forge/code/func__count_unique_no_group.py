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
def _count_unique_no_group(self, df: Any, cols: str, func: AggFunctionSpec) -> Any:
    assert_or_throw('count' == func.name.lower() and func.unique, ValueError(f"{func} can't be applied on {cols}"))
    if cols == '*':
        dkeys = set(df.columns)
    else:
        dkeys = set(cols.split(','))
    return df[list(dkeys)].drop_duplicates().shape[0]