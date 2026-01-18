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
@rename.candidate(lambda df, *args, **kwargs: isinstance(df, rd.Dataset))
def _rename_ray_dataframe(df: rd.Dataset, columns: Dict[str, Any]) -> rd.Dataset:
    if len(columns) == 0:
        return df
    cols = _get_ray_dataframe_columns(df)
    missing = set(columns.keys()) - set(cols)
    if len(missing) > 0:
        raise FugueDataFrameOperationError('found nonexistent columns: {missing}')
    new_cols = [columns.get(name, name) for name in cols]
    return df.map_batches(lambda b: b.rename_columns(new_cols), batch_format='pyarrow', **_ZERO_COPY)