from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
class StataNonCatValueLabel(StataValueLabel):
    """
    Prepare formatted version of value labels

    Parameters
    ----------
    labname : str
        Value label name
    value_labels: Dictionary
        Mapping of values to labels
    encoding : {"latin-1", "utf-8"}
        Encoding to use for value labels.
    """

    def __init__(self, labname: str, value_labels: dict[float, str], encoding: Literal['latin-1', 'utf-8']='latin-1') -> None:
        if encoding not in ('latin-1', 'utf-8'):
            raise ValueError('Only latin-1 and utf-8 are supported.')
        self.labname = labname
        self._encoding = encoding
        self.value_labels = sorted(value_labels.items(), key=lambda x: x[0])
        self._prepare_value_labels()