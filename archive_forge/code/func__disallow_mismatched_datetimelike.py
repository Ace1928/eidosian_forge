from __future__ import annotations
import datetime as dt
import functools
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import (
from pandas._libs.missing import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.compat.numpy import np_version_gt2
from pandas.errors import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
from pandas.io._util import _arrow_dtype_mapping
def _disallow_mismatched_datetimelike(value, dtype: DtypeObj):
    """
    numpy allows np.array(dt64values, dtype="timedelta64[ns]") and
    vice-versa, but we do not want to allow this, so we need to
    check explicitly
    """
    vdtype = getattr(value, 'dtype', None)
    if vdtype is None:
        return
    elif vdtype.kind == 'm' and dtype.kind == 'M' or (vdtype.kind == 'M' and dtype.kind == 'm'):
        raise TypeError(f'Cannot cast {repr(value)} to {dtype}')