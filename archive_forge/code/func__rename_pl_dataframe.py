from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import polars as pl
import pyarrow as pa
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
from fugue import ArrowDataFrame
from fugue.api import (
from fugue.dataframe.dataframe import DataFrame, LocalBoundedDataFrame, _input_schema
from fugue.dataframe.utils import (
from fugue.dataset.api import (
from fugue.exceptions import FugueDataFrameOperationError
from ._utils import build_empty_pl
@rename.candidate(lambda df, *args, **kwargs: isinstance(df, pl.DataFrame))
def _rename_pl_dataframe(df: pl.DataFrame, columns: Dict[str, Any]) -> pl.DataFrame:
    if len(columns) == 0:
        return df
    assert_or_throw(set(columns.keys()).issubset(set(df.columns)), FugueDataFrameOperationError(f'invalid {columns}'))
    return df.rename(columns)