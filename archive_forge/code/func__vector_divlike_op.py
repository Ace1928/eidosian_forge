from __future__ import annotations
from datetime import timedelta
import operator
from typing import (
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.timedeltas import (
from pandas.compat.numpy import function as nv
from pandas.util._validators import validate_endpoints
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.core.ops.common import unpack_zerodim_and_defer
import textwrap
def _vector_divlike_op(self, other, op) -> np.ndarray | Self:
    """
        Shared logic for __truediv__, __floordiv__, and their reversed versions
        with timedelta64-dtype ndarray other.
        """
    result = op(self._ndarray, np.asarray(other))
    if (is_integer_dtype(other.dtype) or is_float_dtype(other.dtype)) and op in [operator.truediv, operator.floordiv]:
        return type(self)._simple_new(result, dtype=result.dtype)
    if op in [operator.floordiv, roperator.rfloordiv]:
        mask = self.isna() | isna(other)
        if mask.any():
            result = result.astype(np.float64)
            np.putmask(result, mask, np.nan)
    return result