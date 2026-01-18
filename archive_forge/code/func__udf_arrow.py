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
def _udf_arrow(dfs: Iterable[pa.RecordBatch]) -> Iterable[pa.RecordBatch]:

    def get_dfs() -> Iterable[LocalDataFrame]:
        cursor_set = False
        func: Any = None
        for df in dfs:
            if df.num_rows > 0:
                adf = pa.Table.from_batches([df])
                if func is None:
                    func = get_alter_func(adf.schema, input_schema.pa_schema, safe=False)
                pdf = ArrowDataFrame(func(adf))
                if not cursor_set:
                    cursor.set(lambda: pdf.peek_array(), 0, 0)
                    cursor_set = True
                yield pdf
    input_df = IterableArrowDataFrame(get_dfs(), input_schema)
    if input_df.empty:
        yield from output_schema.create_empty_arrow_table().to_batches()
        return
    if on_init_once is not None:
        on_init_once(0, input_df)
    output_df = map_func(cursor, input_df)
    if isinstance(output_df, LocalDataFrameIterableDataFrame):
        for res in output_df.native:
            yield from res.as_arrow().to_batches()
    else:
        yield from output_df.as_arrow().to_batches()