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
@select_columns.candidate(lambda df, *args, **kwargs: isinstance(df, pa.Table))
def _select_pa_columns(df: pa.Table, columns: List[Any]) -> pa.Table:
    if len(columns) == 0:
        raise FugueDataFrameOperationError('must select at least one column')
    _assert_no_missing(df, columns=columns)
    return df.select(columns)