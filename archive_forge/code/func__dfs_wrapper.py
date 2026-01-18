from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
from triad import Schema, assert_or_throw
from triad.utils.iter import EmptyAwareIterable, make_empty_aware
from fugue.exceptions import FugueDataFrameInitError
from .array_dataframe import ArrayDataFrame
from .arrow_dataframe import ArrowDataFrame
from .dataframe import (
from .pandas_dataframe import PandasDataFrame
def _dfs_wrapper(self, dfs: Iterable[DataFrame]) -> Iterable[LocalDataFrame]:
    last_empty: Any = None
    last_schema: Any = None
    yielded = False
    for df in dfs:
        if df.empty:
            last_empty = df
        else:
            assert_or_throw(last_schema is None or df.schema == last_schema, lambda: FugueDataFrameInitError(f"encountered schema {df.schema} doesn't match the original schema {df.schema}"))
            if last_schema is None:
                last_schema = df.schema
            yield df.as_local()
            yielded = True
    if not yielded and last_empty is not None:
        yield last_empty