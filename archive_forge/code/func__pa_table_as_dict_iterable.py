from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import cast_pa_table, pa_table_to_pandas
from fugue.dataset.api import (
from fugue.exceptions import FugueDataFrameOperationError
from .api import (
from .dataframe import DataFrame, LocalBoundedDataFrame, _input_schema
from .utils import (
@as_dict_iterable.candidate(lambda df, *args, **kwargs: isinstance(df, pa.Table))
def _pa_table_as_dict_iterable(df: pa.Table, columns: Optional[List[str]]=None) -> Iterable[Dict[str, Any]]:
    yield from pa_table_as_dict_iterable(df, columns=columns)