import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union
import dask.dataframe as dd
import pandas as pd
from distributed import Client
from triad.collections import Schema
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.pandas_like import PandasUtils
from triad.utils.threading import RunOnce
from triad.utils.io import makedirs
from fugue import StructuredRawSQL
from fugue.collections.partition import (
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue.dataframe import (
from fugue.dataframe.utils import get_join_schemas
from fugue.exceptions import FugueBug
from fugue.execution.execution_engine import ExecutionEngine, MapEngine, SQLEngine
from fugue.execution.native_execution_engine import NativeExecutionEngine
from fugue_dask._constants import FUGUE_DASK_DEFAULT_CONF
from fugue_dask._io import load_df, save_df
from fugue_dask._utils import (
from fugue_dask.dataframe import DaskDataFrame
from ._constants import FUGUE_DASK_USE_ARROW
class DaskSQLEngine(SQLEngine):
    """Dask-sql implementation."""

    @property
    def dialect(self) -> Optional[str]:
        return 'trino'

    def to_df(self, df: AnyDataFrame, schema: Any=None) -> DataFrame:
        return to_dask_engine_df(df, schema)

    @property
    def is_distributed(self) -> bool:
        return True

    def select(self, dfs: DataFrames, statement: StructuredRawSQL) -> DataFrame:
        try:
            from dask_sql import Context
        except ImportError:
            raise ImportError('dask-sql is not installed. Please install it with `pip install dask-sql`')
        ctx = Context()
        _dfs: Dict[str, dd.DataFrame] = {k: self._to_safe_df(v) for k, v in dfs.items()}
        sql = statement.construct(dialect=self.dialect, log=self.log)
        res = ctx.sql(sql, dataframes=_dfs, config_options={'sql.identifier.case_sensitive': True})
        return DaskDataFrame(res)

    def _to_safe_df(self, df: DataFrame) -> dd.DataFrame:
        df = self.to_df(df)
        return df.native.astype(df.schema.to_pandas_dtype(use_extension_types=True, use_arrow_dtype=False))