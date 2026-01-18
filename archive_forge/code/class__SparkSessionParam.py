from typing import Any, Optional, Tuple
import pyspark.rdd as pr
import pyspark.sql as ps
from pyspark import SparkContext
from pyspark.sql import SparkSession
from triad import run_at_def
from fugue import DataFrame, ExecutionEngine, register_execution_engine
from fugue.dev import (
from fugue.extensions import namespace_candidate
from fugue.plugins import as_fugue_dataset, infer_execution_engine, parse_creator
from fugue_spark.dataframe import SparkDataFrame
from fugue_spark.execution_engine import SparkExecutionEngine
from ._utils.misc import SparkConnectDataFrame, SparkConnectSession, is_spark_dataframe
@fugue_annotated_param(SparkSession)
class _SparkSessionParam(ExecutionEngineParam):

    def to_input(self, engine: ExecutionEngine) -> Any:
        assert isinstance(engine, SparkExecutionEngine)
        return engine.spark_session