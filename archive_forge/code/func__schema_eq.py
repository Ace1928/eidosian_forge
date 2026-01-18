import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from fsspec import AbstractFileSystem
from triad import Schema, assert_or_throw
from triad.collections.schema import SchemaError
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.io import url_to_fs
from triad.utils.pyarrow import pa_batch_to_dicts
from .api import as_fugue_df, get_column_names, normalize_column_names, rename
from .dataframe import DataFrame, LocalBoundedDataFrame
def _schema_eq(s1: Schema, s2: Schema) -> bool:
    if s1 == s2:
        return True
    return s1.names == s2.names and all((_pa_type_eq(f1.type, f2.type) for f1, f2 in zip(s1.fields, s2.fields)))