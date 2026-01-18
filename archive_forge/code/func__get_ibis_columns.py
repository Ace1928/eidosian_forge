from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
from triad import Schema, assert_or_throw
from fugue import DataFrame, IterableDataFrame, LocalBoundedDataFrame
from fugue.dataframe.dataframe import _input_schema
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
from fugue.plugins import drop_columns, get_column_names, is_df, rename
from ._compat import IbisSchema, IbisTable
from ._utils import pa_to_ibis_type, to_schema
@get_column_names.candidate(lambda df: isinstance(df, IbisTable))
def _get_ibis_columns(df: IbisTable) -> List[Any]:
    return df.columns