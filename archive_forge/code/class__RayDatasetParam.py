from typing import Any
import ray.data as rd
from triad import run_at_def
from fugue import DataFrame, register_execution_engine
from fugue.dev import (
from fugue.plugins import as_fugue_dataset, infer_execution_engine
from .dataframe import RayDataFrame
from .execution_engine import RayExecutionEngine
@fugue_annotated_param(rd.Dataset)
class _RayDatasetParam(DataFrameParam):

    def to_input_data(self, df: DataFrame, ctx: Any) -> Any:
        assert isinstance(ctx, RayExecutionEngine)
        return ctx._to_ray_df(df).native

    def to_output_df(self, output: Any, schema: Any, ctx: Any) -> DataFrame:
        assert isinstance(output, rd.Dataset)
        assert isinstance(ctx, RayExecutionEngine)
        return RayDataFrame(output, schema=schema)

    def count(self, df: DataFrame) -> int:
        raise NotImplementedError('not allowed')