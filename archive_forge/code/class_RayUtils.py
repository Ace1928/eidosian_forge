from typing import Any, Callable, Dict, List, Tuple, Union
import modin.pandas as pd
import numpy
import pandas
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd import QPDEngine, run_sql
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_pandas.engine import _RowsIndexer
class RayUtils(PandasLikeUtils[pd.DataFrame, pd.Series]):

    def drop_duplicates(self, ndf: pd.DataFrame) -> pd.DataFrame:
        keys = list(ndf.columns)
        if ndf.empty:
            return ndf
        if not isinstance(ndf.index, pd.RangeIndex):
            ndf = ndf.reset_index(drop=True)
        nulldf = ndf[keys].isnull().rename({x: x + '_ddn' for x in keys})
        filled = ndf[keys].fillna(0).rename({x: x + '_ddg' for x in keys})
        tp = pd.concat([filled, nulldf], axis=1).apply(tuple, axis=1)
        tp = tp.drop_duplicates().isnull().rename('__dedup__')
        ndf = ndf.merge(tp, how='inner', left_index=True, right_index=True)
        return ndf[keys]