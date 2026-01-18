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
def _adjust_df(res: pd.DataFrame, as_fugue: bool):
    return res if not as_fugue else PandasDataFrame(res)