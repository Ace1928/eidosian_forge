from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
import pyspark.sql as ps
from pyspark.sql.functions import col
from triad import SerializableRLock
from triad.collections.schema import SchemaError
from triad.utils.assertion import assert_or_throw
from fugue.dataframe import (
from fugue.dataframe.utils import pa_table_as_array, pa_table_as_dicts
from fugue.exceptions import FugueDataFrameOperationError
from fugue.plugins import (
from ._utils.convert import (
from ._utils.misc import is_spark_connect, is_spark_dataframe
@select_columns.candidate(lambda df, *args, **kwargs: is_spark_dataframe(df))
def _select_spark_df_columns(df: ps.DataFrame, columns: List[Any], as_fugue: bool=False) -> Any:
    if len(columns) == 0:
        raise FugueDataFrameOperationError('must select at least one column')
    _assert_no_missing(df, columns)
    return _adjust_df(df[columns], as_fugue=as_fugue)