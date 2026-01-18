import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _from_cupy_array(data: DataType, missing: FloatCompatible, nthread: int, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes]) -> DispatchedDataBackendReturnType:
    """Initialize DMatrix from cupy ndarray."""
    data = _transform_cupy_array(data)
    interface_str = _cuda_array_interface(data)
    handle = ctypes.c_void_p()
    config = bytes(json.dumps({'missing': missing, 'nthread': nthread}), 'utf-8')
    _check_call(_LIB.XGDMatrixCreateFromCudaArrayInterface(interface_str, config, ctypes.byref(handle)))
    return (handle, feature_names, feature_types)