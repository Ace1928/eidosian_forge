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
class PythonFormatter(Formatter[Mapping, list, Mapping]):

    def __init__(self, features=None, lazy=False):
        super().__init__(features)
        self.lazy = lazy

    def format_row(self, pa_table: pa.Table) -> Mapping:
        if self.lazy:
            return LazyRow(pa_table, self)
        row = self.python_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        return row

    def format_column(self, pa_table: pa.Table) -> list:
        column = self.python_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        return column

    def format_batch(self, pa_table: pa.Table) -> Mapping:
        if self.lazy:
            return LazyBatch(pa_table, self)
        batch = self.python_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        return batch