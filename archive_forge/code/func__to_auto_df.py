from typing import Any, Optional, Union
import dask.dataframe as dd
import duckdb
import pyarrow as pa
from dask.distributed import Client
from duckdb import DuckDBPyConnection
from triad import assert_or_throw
from fugue import DataFrame, MapEngine, PartitionSpec
from fugue_dask import DaskDataFrame, DaskExecutionEngine
from fugue_dask.execution_engine import DaskMapEngine
from .dataframe import DuckDataFrame
from .execution_engine import DuckExecutionEngine, _to_duck_df
def _to_auto_df(self, df: Any, schema: Any=None) -> Union[DuckDataFrame, DaskDataFrame]:
    if isinstance(df, (DuckDataFrame, DaskDataFrame)):
        assert_or_throw(schema is None, ValueError('schema must be None when df is a DataFrame'))
        return df
    if isinstance(df, dd.DataFrame):
        return self._dask_engine.to_df(df, schema)
    return _to_duck_df(self, df, schema)