from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
from duckdb import DuckDBPyRelation
from triad import Schema, assert_or_throw
from triad.utils.pyarrow import LARGE_TYPES_REPLACEMENT, replace_types_in_table
from fugue import ArrowDataFrame, DataFrame, LocalBoundedDataFrame
from fugue.dataframe.arrow_dataframe import _pa_table_as_pandas
from fugue.dataframe.utils import (
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
from fugue.plugins import (
from ._utils import encode_column_name, to_duck_type, to_pa_type
@get_num_partitions.candidate(lambda df: isinstance(df, DuckDBPyRelation))
def _duckdb_num_partitions(df: DuckDBPyRelation) -> int:
    return 1