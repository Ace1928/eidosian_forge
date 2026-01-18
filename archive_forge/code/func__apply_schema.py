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
def _apply_schema(self, pdf: pd.DataFrame, schema: Optional[Schema]) -> Tuple[pd.DataFrame, Schema]:
    PD_UTILS.ensure_compatible(pdf)
    pschema = _input_schema(pdf)
    if schema is None or pschema == schema:
        return (pdf, pschema.assert_not_empty())
    pdf = pdf[schema.assert_not_empty().names]
    return (PD_UTILS.cast_df(pdf, schema.pa_schema, use_extension_types=True, use_arrow_dtype=False), schema)