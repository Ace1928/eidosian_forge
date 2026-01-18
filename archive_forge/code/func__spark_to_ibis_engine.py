from typing import Any, Callable
import ibis
from pyspark.sql import DataFrame as PySparkDataFrame
from triad.utils.assertion import assert_or_throw
from fugue import DataFrame, DataFrames, ExecutionEngine
from fugue_ibis import IbisTable
from fugue_ibis._utils import to_schema
from fugue_ibis.execution.ibis_engine import IbisEngine, parse_ibis_engine
from fugue_spark.dataframe import SparkDataFrame
from fugue_spark.execution_engine import SparkExecutionEngine
@parse_ibis_engine.candidate(lambda obj, *args, **kwargs: isinstance(obj, SparkExecutionEngine))
def _spark_to_ibis_engine(obj: Any, engine: ExecutionEngine) -> IbisEngine:
    return SparkIbisEngine(engine)