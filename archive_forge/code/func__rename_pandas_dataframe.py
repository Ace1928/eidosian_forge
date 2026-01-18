from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from triad import assert_or_throw
from triad.collections.schema import Schema
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import pa_batch_to_dicts
from fugue.dataset.api import (
from fugue.exceptions import FugueDataFrameOperationError
from .api import (
from .dataframe import DataFrame, LocalBoundedDataFrame, _input_schema
@rename.candidate(lambda df, *args, **kwargs: isinstance(df, pd.DataFrame))
def _rename_pandas_dataframe(df: pd.DataFrame, columns: Dict[str, Any], as_fugue: bool=False) -> Any:
    if len(columns) == 0:
        return df
    _assert_no_missing(df, columns.keys())
    return _adjust_df(df.rename(columns=columns), as_fugue=as_fugue)