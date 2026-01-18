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
def _to_dicts(df: pd.DataFrame, columns: Optional[List[str]]=None, schema: Optional[Schema]=None) -> Iterable[List[Dict[str, Any]]]:
    cols = list(df.columns) if columns is None else columns
    assert_or_throw(len(cols) > 0, ValueError('columns cannot be empty'))
    pa_schema = schema.extract(cols).pa_schema if schema is not None else None
    adf = PD_UTILS.as_arrow(df[cols], schema=pa_schema)
    for batch in adf.to_batches():
        if batch.num_rows > 0:
            yield pa_batch_to_dicts(batch)