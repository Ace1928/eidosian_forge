from collections.abc import Mapping, MutableMapping
from functools import partial
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from .. import config
from ..features import Features
from ..features.features import _ArrayXDExtensionType, _is_zero_copy_only, decode_nested_example, pandas_types_mapper
from ..table import Table
from ..utils.py_utils import no_op_if_value_is_null
class PandasFormatter(Formatter[pd.DataFrame, pd.Series, pd.DataFrame]):

    def format_row(self, pa_table: pa.Table) -> pd.DataFrame:
        row = self.pandas_arrow_extractor().extract_row(pa_table)
        row = self.pandas_features_decoder.decode_row(row)
        return row

    def format_column(self, pa_table: pa.Table) -> pd.Series:
        column = self.pandas_arrow_extractor().extract_column(pa_table)
        column = self.pandas_features_decoder.decode_column(column, pa_table.column_names[0])
        return column

    def format_batch(self, pa_table: pa.Table) -> pd.DataFrame:
        row = self.pandas_arrow_extractor().extract_batch(pa_table)
        row = self.pandas_features_decoder.decode_batch(row)
        return row