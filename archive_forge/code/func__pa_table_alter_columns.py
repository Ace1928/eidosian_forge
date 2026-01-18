from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import cast_pa_table, pa_table_to_pandas
from fugue.dataset.api import (
from fugue.exceptions import FugueDataFrameOperationError
from .api import (
from .dataframe import DataFrame, LocalBoundedDataFrame, _input_schema
from .utils import (
@alter_columns.candidate(lambda df, *args, **kwargs: isinstance(df, pa.Table))
def _pa_table_alter_columns(df: pa.Table, columns: Any, as_fugue: bool=False) -> pa.Table:
    schema = Schema(df.schema)
    new_schema = schema.alter(columns)
    if schema != new_schema:
        df = cast_pa_table(df, new_schema.pa_schema)
    return df if not as_fugue else ArrowDataFrame(df)