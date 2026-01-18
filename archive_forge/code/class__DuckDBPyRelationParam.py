from typing import Any
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from triad import run_at_def
from fugue import (
from fugue.dev import (
from fugue.plugins import infer_execution_engine
from fugue_duckdb.dataframe import DuckDataFrame
from fugue_duckdb.execution_engine import DuckDBEngine, DuckExecutionEngine
@fugue_annotated_param(DuckDBPyRelation)
class _DuckDBPyRelationParam(DataFrameParam):

    def to_input_data(self, df: DataFrame, ctx: Any) -> Any:
        assert isinstance(ctx, DuckExecutionEngine)
        return ctx.to_df(df).native

    def to_output_df(self, output: Any, schema: Any, ctx: Any) -> DataFrame:
        assert isinstance(output, DuckDBPyRelation)
        assert isinstance(ctx, DuckExecutionEngine)
        return DuckDataFrame(output)

    def count(self, df: Any) -> int:
        return df.count()