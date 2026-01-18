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
def _to_dask_df(self, df: Any, schema: Any=None) -> DaskDataFrame:
    if isinstance(df, DuckDataFrame):
        res = self._dask_engine.to_df(df.as_pandas(), df.schema)
        res.reset_metadata(df.metadata if df.has_metadata else None)
        return res
    return self._dask_engine.to_df(df, schema)