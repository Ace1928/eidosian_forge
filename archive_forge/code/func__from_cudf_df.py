import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _from_cudf_df(data: DataType, missing: FloatCompatible, nthread: int, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes], enable_categorical: bool) -> DispatchedDataBackendReturnType:
    data, cat_codes, feature_names, feature_types = _transform_cudf_df(data, feature_names, feature_types, enable_categorical)
    interfaces_str = _cudf_array_interfaces(data, cat_codes)
    handle = ctypes.c_void_p()
    config = bytes(json.dumps({'missing': missing, 'nthread': nthread}), 'utf-8')
    _check_call(_LIB.XGDMatrixCreateFromCudaColumnar(interfaces_str, config, ctypes.byref(handle)))
    return (handle, feature_names, feature_types)