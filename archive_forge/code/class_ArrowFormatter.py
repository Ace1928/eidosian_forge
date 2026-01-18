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
class ArrowFormatter(Formatter[pa.Table, pa.Array, pa.Table]):

    def format_row(self, pa_table: pa.Table) -> pa.Table:
        return self.simple_arrow_extractor().extract_row(pa_table)

    def format_column(self, pa_table: pa.Table) -> pa.Array:
        return self.simple_arrow_extractor().extract_column(pa_table)

    def format_batch(self, pa_table: pa.Table) -> pa.Table:
        return self.simple_arrow_extractor().extract_batch(pa_table)