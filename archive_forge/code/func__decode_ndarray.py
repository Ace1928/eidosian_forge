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
def _decode_ndarray(self, obj: NDArrayRep) -> npt.NDArray[Any]:
    array = obj['array']
    order = obj['order']
    dtype = obj['dtype']
    shape = obj['shape']
    decoded = self._decode(array)
    ndarray: npt.NDArray[Any]
    if isinstance(decoded, bytes):
        ndarray = np.copy(np.frombuffer(decoded, dtype=dtype))
        if order != sys.byteorder:
            ndarray.byteswap(inplace=True)
    else:
        ndarray = np.array(decoded, dtype=dtype)
    if len(shape) > 1:
        ndarray = ndarray.reshape(shape)
    return ndarray