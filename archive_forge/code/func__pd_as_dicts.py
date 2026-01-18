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
@as_dicts.candidate(lambda df, *args, **kwargs: isinstance(df, pd.DataFrame))
def _pd_as_dicts(df: pd.DataFrame, columns: Optional[List[str]]=None) -> List[Dict[str, Any]]:
    res: List[Dict[str, Any]] = []
    for block in _to_dicts(df, columns):
        res += block
    return res