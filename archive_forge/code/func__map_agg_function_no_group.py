from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasUtils
from qpd import QPDEngine, run_sql
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
def _map_agg_function_no_group(self, func: AggFunctionSpec) -> Any:
    name = func.name.lower()
    if name == 'sum':
        if not func.unique:
            return lambda x: x.sum(min_count=1)
        else:
            return lambda x: x.drop_duplicates().sum(min_count=1)
    if name in ['avg', 'mean']:
        if not func.unique:
            return 'mean'
        else:
            return lambda x: x.drop_duplicates().mean()
    if name in ['first', 'first_value']:
        if func.dropna:
            return lambda x: x[x.first_valid_index()]
        else:
            return lambda x: x.head(1)
    if name in ['last', 'last_value']:
        if func.dropna:
            return lambda x: x[x.last_valid_index()]
        else:
            return lambda x: x.tail(1)
    if name == 'count':
        if not func.unique and (not func.dropna):
            return 'size'
        if func.unique and (not func.dropna):
            return lambda x: x.drop_duplicates().size
        if func.unique and func.dropna:
            return 'nunique'
        if not func.unique and func.dropna:
            return 'count'
    if name == 'min':
        return lambda x: x.dropna().min()
    if name == 'max':
        return lambda x: x.dropna().max()
    return name