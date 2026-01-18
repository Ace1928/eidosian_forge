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
def _prepare_value_labels(self) -> None:
    """Encode value labels."""
    self.text_len = 0
    self.txt: list[bytes] = []
    self.n = 0
    self.off = np.array([], dtype=np.int32)
    self.val = np.array([], dtype=np.int32)
    self.len = 0
    offsets: list[int] = []
    values: list[float] = []
    for vl in self.value_labels:
        category: str | bytes = vl[1]
        if not isinstance(category, str):
            category = str(category)
            warnings.warn(value_label_mismatch_doc.format(self.labname), ValueLabelTypeMismatch, stacklevel=find_stack_level())
        category = category.encode(self._encoding)
        offsets.append(self.text_len)
        self.text_len += len(category) + 1
        values.append(vl[0])
        self.txt.append(category)
        self.n += 1
    if self.text_len > 32000:
        raise ValueError('Stata value labels for a single variable must have a combined length less than 32,000 characters.')
    self.off = np.array(offsets, dtype=np.int32)
    self.val = np.array(values, dtype=np.int32)
    self.len = 4 + 4 + 4 * self.n + 4 * self.n + self.text_len