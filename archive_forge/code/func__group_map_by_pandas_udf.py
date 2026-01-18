import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from uuid import uuid4
import pandas as pd
import pyarrow as pa
import pyspark.sql as ps
from py4j.protocol import Py4JError
from pyspark import StorageLevel
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast, col, lit, row_number
from pyspark.sql.window import Window
from triad import IndexedOrderedDict, ParamDict, Schema, SerializableRLock
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.iter import EmptyAwareIterable
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import get_alter_func
from triad.utils.threading import RunOnce
from fugue import StructuredRawSQL
from fugue.collections.partition import (
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue.dataframe import (
from fugue.dataframe.utils import get_join_schemas
from fugue.exceptions import FugueDataFrameInitError
from fugue.execution.execution_engine import ExecutionEngine, MapEngine, SQLEngine
from ._constants import FUGUE_SPARK_CONF_USE_PANDAS_UDF, FUGUE_SPARK_DEFAULT_CONF
from ._utils.convert import (
from ._utils.io import SparkIO
from ._utils.misc import is_spark_connect as _is_spark_connect
from ._utils.misc import is_spark_dataframe
from ._utils.partition import even_repartition, hash_repartition, rand_repartition
from .dataframe import SparkDataFrame
def _group_map_by_pandas_udf(self, df: DataFrame, map_func: Callable[[PartitionCursor, LocalDataFrame], LocalDataFrame], output_schema: Any, partition_spec: PartitionSpec, on_init: Optional[Callable[[int, DataFrame], Any]]=None, map_func_format_hint: Optional[str]=None) -> DataFrame:
    presort = partition_spec.presort
    presort_keys = list(presort.keys())
    presort_asc = list(presort.values())
    output_schema = Schema(output_schema)
    input_schema = df.schema
    cursor = partition_spec.get_cursor(input_schema, 0)
    on_init_once: Any = None if on_init is None else RunOnce(on_init, lambda *args, **kwargs: to_uuid(id(on_init), id(args[0])))

    def _udf_pandas(pdf: Any) -> pd.DataFrame:
        if pdf.shape[0] == 0:
            return _to_safe_spark_worker_pandas(PandasDataFrame(schema=output_schema).as_pandas())
        if len(partition_spec.presort) > 0:
            pdf = pdf.sort_values(presort_keys, ascending=presort_asc)
        input_df = PandasDataFrame(pdf.reset_index(drop=True), input_schema, pandas_df_wrapper=True)
        if on_init_once is not None:
            on_init_once(0, input_df)
        cursor.set(lambda: input_df.peek_array(), 0, 0)
        output_df = map_func(cursor, input_df)
        return _to_safe_spark_worker_pandas(output_df.as_pandas())
    df = self.to_df(df)
    gdf = df.native.groupBy(*partition_spec.partition_by)
    sdf = gdf.applyInPandas(_udf_pandas, schema=to_spark_schema(output_schema))
    return SparkDataFrame(sdf)