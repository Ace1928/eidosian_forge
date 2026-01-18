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
def _set_formats_and_types(self, dtypes: Series) -> None:
    self.typlist = []
    self.fmtlist = []
    for col, dtype in dtypes.items():
        force_strl = col in self._convert_strl
        fmt = _dtype_to_default_stata_fmt(dtype, self.data[col], dta_version=self._dta_version, force_strl=force_strl)
        self.fmtlist.append(fmt)
        self.typlist.append(_dtype_to_stata_type_117(dtype, self.data[col], force_strl))