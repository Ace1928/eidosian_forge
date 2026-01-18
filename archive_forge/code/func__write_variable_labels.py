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
def _write_variable_labels(self) -> None:
    self._update_map('variable_labels')
    bio = BytesIO()
    vl_len = 80 if self._dta_version == 117 else 320
    blank = _pad_bytes_new('', vl_len + 1)
    if self._variable_labels is None:
        for _ in range(self.nvar):
            bio.write(blank)
        self._write_bytes(self._tag(bio.getvalue(), 'variable_labels'))
        return
    for col in self.data:
        if col in self._variable_labels:
            label = self._variable_labels[col]
            if len(label) > 80:
                raise ValueError('Variable labels must be 80 characters or fewer')
            try:
                encoded = label.encode(self._encoding)
            except UnicodeEncodeError as err:
                raise ValueError(f'Variable labels must contain only characters that can be encoded in {self._encoding}') from err
            bio.write(_pad_bytes_new(encoded, vl_len + 1))
        else:
            bio.write(blank)
    self._write_bytes(self._tag(bio.getvalue(), 'variable_labels'))