from typing import Any, Callable, Dict, List, Optional, Type, Union
import pyarrow as pa
import ray
from duckdb import DuckDBPyConnection
from packaging import version
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.threading import RunOnce
from fugue import (
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue_duckdb.dataframe import DuckDataFrame
from fugue_duckdb.execution_engine import DuckExecutionEngine
from ._constants import FUGUE_RAY_DEFAULT_BATCH_SIZE, FUGUE_RAY_ZERO_COPY
from ._utils.cluster import get_default_partitions, get_default_shuffle_partitions
from ._utils.dataframe import add_coarse_partition_key, add_partition_key
from ._utils.io import RayIO
from .dataframe import RayDataFrame
def _udf(adf: pa.Table) -> pa.Table:
    if adf.shape[0] == 0:
        return output_schema.create_empty_arrow_table()
    input_df = ArrowDataFrame(adf)
    if on_init_once is not None:
        on_init_once(0, input_df)
    cursor.set(lambda: input_df.peek_array(), 0, 0)
    output_df = map_func(cursor, input_df)
    return output_df.as_arrow()