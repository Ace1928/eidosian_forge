from __future__ import annotations
from functools import wraps
from typing import (
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
def _fillna_prep(values, mask: npt.NDArray[np.bool_] | None=None) -> npt.NDArray[np.bool_]:
    if mask is None:
        mask = isna(values)
    return mask