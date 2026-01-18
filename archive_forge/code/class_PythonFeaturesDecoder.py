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
class PythonFeaturesDecoder:

    def __init__(self, features: Optional[Features]):
        self.features = features

    def decode_row(self, row: dict) -> dict:
        return self.features.decode_example(row) if self.features else row

    def decode_column(self, column: list, column_name: str) -> list:
        return self.features.decode_column(column, column_name) if self.features else column

    def decode_batch(self, batch: dict) -> dict:
        return self.features.decode_batch(batch) if self.features else batch