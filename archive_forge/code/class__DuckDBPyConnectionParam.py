from typing import Any
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from triad import run_at_def
from fugue import (
from fugue.dev import (
from fugue.plugins import infer_execution_engine
from fugue_duckdb.dataframe import DuckDataFrame
from fugue_duckdb.execution_engine import DuckDBEngine, DuckExecutionEngine
@fugue_annotated_param(DuckDBPyConnection)
class _DuckDBPyConnectionParam(ExecutionEngineParam):

    def to_input(self, engine: ExecutionEngine) -> Any:
        assert isinstance(engine, DuckExecutionEngine)
        return engine.connection