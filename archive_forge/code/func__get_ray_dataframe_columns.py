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
@get_column_names.candidate(lambda df: isinstance(df, rd.Dataset))
def _get_ray_dataframe_columns(df: rd.Dataset) -> List[Any]:
    if hasattr(df, 'columns'):
        return df.columns(fetch_if_missing=True)
    else:
        fmt, _ = get_dataset_format(df)
        if fmt == 'pandas':
            return list(df.schema(True).names)
        elif fmt == 'arrow':
            return df.schema(fetch_if_missing=True).names
        raise NotImplementedError(f'{fmt} is not supported')