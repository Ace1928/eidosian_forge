from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
import ray
import ray.data as rd
from triad import assert_or_throw
from triad.collections.schema import Schema
from triad.utils.pyarrow import cast_pa_table
from fugue.dataframe import ArrowDataFrame, DataFrame, LocalBoundedDataFrame
from fugue.dataframe.dataframe import _input_schema
from fugue.dataframe.utils import pa_table_as_array, pa_table_as_dicts
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
from fugue.plugins import (
from ._constants import _ZERO_COPY
from ._utils.dataframe import build_empty, get_dataset_format, materialize, to_schema
@as_dicts.candidate(lambda df, *args, **kwargs: isinstance(df, rd.Dataset))
def _rd_as_dicts(df: rd.Dataset, columns: Optional[List[str]]=None, type_safe: bool=False) -> List[Dict[str, Any]]:
    assert_or_throw(columns is None or len(columns) > 0, ValueError('empty columns'))
    _df = df if columns is None or len(columns) == 0 else df.select_columns(columns)
    adf = _rd_as_arrow(_df)
    return pa_table_as_dicts(adf)