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
@as_arrow.candidate(lambda df: isinstance(df, pl.DataFrame))
def _pl_as_arrow(df: pl.DataFrame) -> pa.Table:
    adf = df.to_arrow()
    adf = replace_types_in_table(adf, LARGE_TYPES_REPLACEMENT)
    return adf