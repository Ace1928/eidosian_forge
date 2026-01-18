from typing import Any, Dict, Iterable, List, Optional, Tuple
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from triad import assert_or_throw
from triad.collections.schema import Schema
from triad.utils.assertion import assert_arg_not_none
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import cast_pa_table
from fugue.dataframe import DataFrame, LocalBoundedDataFrame, PandasDataFrame
from fugue.dataframe.dataframe import _input_schema
from fugue.dataframe.pandas_dataframe import _pd_as_dicts
from fugue.exceptions import FugueDataFrameOperationError
from fugue.plugins import (
from ._constants import FUGUE_DASK_USE_ARROW
from ._utils import DASK_UTILS, collect, get_default_partitions
def _to_array_chunks(df: dd.DataFrame, columns: Optional[List[str]]=None, type_safe: bool=False, schema: Optional[Schema]=None) -> Tuple[List[Any]]:
    assert_or_throw(columns is None or len(columns) > 0, ValueError('empty columns'))
    _df = df if columns is None or len(columns) == 0 else df[columns]

    def _to_list(pdf: pd.DataFrame) -> List[Any]:
        return list(PD_UTILS.as_array_iterable(pdf, schema=None if schema is None else schema.pa_schema, columns=columns, type_safe=type_safe))
    return collect(_df, _to_list)