from typing import Any
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from triad import run_at_def
from fugue import (
from fugue.dev import (
from fugue.plugins import infer_execution_engine
from fugue_duckdb.dataframe import DuckDataFrame
from fugue_duckdb.execution_engine import DuckDBEngine, DuckExecutionEngine
@infer_execution_engine.candidate(lambda objs: is_pandas_or(objs, (DuckDBPyRelation, DuckDataFrame)))
def _infer_duckdb_client(objs: Any) -> Any:
    return 'duckdb'