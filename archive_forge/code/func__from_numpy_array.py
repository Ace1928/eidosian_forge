import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _from_numpy_array(data: DataType, missing: FloatCompatible, nthread: int, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes], data_split_mode: DataSplitMode=DataSplitMode.ROW) -> DispatchedDataBackendReturnType:
    """Initialize data from a 2-D numpy matrix."""
    _check_data_shape(data)
    data, _ = _ensure_np_dtype(data, data.dtype)
    handle = ctypes.c_void_p()
    _check_call(_LIB.XGDMatrixCreateFromDense(_array_interface(data), make_jcargs(missing=float(missing), nthread=int(nthread), data_split_mode=int(data_split_mode)), ctypes.byref(handle)))
    return (handle, feature_names, feature_types)