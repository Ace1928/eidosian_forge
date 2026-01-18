import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _from_pandas_series(data: DataType, missing: FloatCompatible, nthread: int, enable_categorical: bool, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes]) -> DispatchedDataBackendReturnType:
    if data.dtype.name not in _pandas_dtype_mapper and (not (is_pd_cat_dtype(data.dtype) and enable_categorical)):
        _invalid_dataframe_dtype(data)
    if enable_categorical and is_pd_cat_dtype(data.dtype):
        data = data.cat.codes
    return _from_numpy_array(data.values.reshape(data.shape[0], 1).astype('float'), missing, nthread, feature_names, feature_types)