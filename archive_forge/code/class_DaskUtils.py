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
class DaskUtils(PandasLikeUtils[pd.DataFrame, pd.Series]):

    def concat_dfs(self, *dfs: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(list(dfs))