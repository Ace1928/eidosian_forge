from __future__ import annotations
import logging # isort:skip
import base64
import datetime as dt
import sys
from array import array as TypedArray
from math import isinf, isnan
from types import SimpleNamespace
from typing import (
import numpy as np
from ..util.dataclasses import (
from ..util.dependencies import uses_pandas
from ..util.serialization import (
from ..util.warnings import BokehUserWarning, warn
from .types import ID
def _encode_other(self, obj: Any) -> AnyRep:
    if is_datetime_type(obj):
        return convert_datetime_type(obj)
    if is_timedelta_type(obj):
        return convert_timedelta_type(obj)
    if isinstance(obj, dt.date):
        return obj.isoformat()
    if np.issubdtype(type(obj), np.floating):
        return self._encode_float(float(obj))
    if np.issubdtype(type(obj), np.integer):
        return self._encode_int(int(obj))
    if np.issubdtype(type(obj), np.bool_):
        return self._encode_bool(bool(obj))
    if uses_pandas(obj):
        import pandas as pd
        if isinstance(obj, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray)):
            return self._encode_ndarray(transform_series(obj))
        elif obj is pd.NA:
            return None
    if hasattr(obj, '__array__') and isinstance((arr := obj.__array__()), np.ndarray):
        return self._encode_ndarray(arr)
    self.error(f"can't serialize {type(obj)}")