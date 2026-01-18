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
Iterable of dataframes