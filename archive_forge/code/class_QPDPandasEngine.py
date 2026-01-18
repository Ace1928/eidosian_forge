import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union
import numpy as np
import pandas as pd
from triad import Schema
from triad.collections.dict import IndexedOrderedDict
from triad.utils.assertion import assert_or_throw
from triad.utils.io import makedirs
from triad.utils.pandas_like import PandasUtils
from fugue._utils.io import load_df, save_df
from fugue._utils.misc import import_fsql_dependency
from fugue.collections.partition import (
from fugue.collections.sql import StructuredRawSQL
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue.dataframe import (
from fugue.dataframe.dataframe import as_fugue_df
from fugue.dataframe.utils import get_join_schemas
from .execution_engine import (
class QPDPandasEngine(SQLEngine):
    """QPD execution implementation.

    :param execution_engine: the execution engine this sql engine will run on
    """

    @property
    def dialect(self) -> Optional[str]:
        return 'spark'

    def to_df(self, df: AnyDataFrame, schema: Any=None) -> DataFrame:
        return _to_native_execution_engine_df(df, schema)

    @property
    def is_distributed(self) -> bool:
        return False

    def select(self, dfs: DataFrames, statement: StructuredRawSQL) -> DataFrame:
        qpd_pandas = import_fsql_dependency('qpd_pandas')
        _dfs, _sql = self.encode(dfs, statement)
        _dd = {k: self.to_df(v).as_pandas() for k, v in _dfs.items()}
        df = qpd_pandas.run_sql_on_pandas(_sql, _dd, ignore_case=True)
        return self.to_df(df)