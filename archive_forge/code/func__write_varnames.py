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
def _write_varnames(self) -> None:
    self._update_map('varnames')
    bio = BytesIO()
    vn_len = 32 if self._dta_version == 117 else 128
    for name in self.varlist:
        name = self._null_terminate_str(name)
        name = _pad_bytes_new(name[:32].encode(self._encoding), vn_len + 1)
        bio.write(name)
    self._write_bytes(self._tag(bio.getvalue(), 'varnames'))