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
def _setup_dtype(self) -> np.dtype:
    """Map between numpy and state dtypes"""
    if self._dtype is not None:
        return self._dtype
    dtypes = []
    for i, typ in enumerate(self._typlist):
        if typ in self.NUMPY_TYPE_MAP:
            typ = cast(str, typ)
            dtypes.append((f's{i}', f'{self._byteorder}{self.NUMPY_TYPE_MAP[typ]}'))
        else:
            dtypes.append((f's{i}', f'S{typ}'))
    self._dtype = np.dtype(dtypes)
    return self._dtype