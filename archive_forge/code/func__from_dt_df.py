import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _from_dt_df(data: DataType, missing: Optional[FloatCompatible], nthread: int, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes], enable_categorical: bool) -> DispatchedDataBackendReturnType:
    if enable_categorical:
        raise ValueError('categorical data in datatable is not supported yet.')
    data, feature_names, feature_types = _transform_dt_df(data, feature_names, feature_types, None, None)
    ptrs = (ctypes.c_void_p * data.ncols)()
    if hasattr(data, 'internal') and hasattr(data.internal, 'column'):
        for icol in range(data.ncols):
            col = data.internal.column(icol)
            ptr = col.data_pointer
            ptrs[icol] = ctypes.c_void_p(ptr)
    else:
        from datatable.internal import frame_column_data_r
        for icol in range(data.ncols):
            ptrs[icol] = frame_column_data_r(data, icol)
    feature_type_strings = (ctypes.c_char_p * data.ncols)()
    for icol in range(data.ncols):
        feature_type_strings[icol] = ctypes.c_char_p(data.stypes[icol].name.encode('utf-8'))
    _warn_unused_missing(data, missing)
    handle = ctypes.c_void_p()
    _check_call(_LIB.XGDMatrixCreateFromDT(ptrs, feature_type_strings, c_bst_ulong(data.shape[0]), c_bst_ulong(data.shape[1]), ctypes.byref(handle), ctypes.c_int(nthread)))
    return (handle, feature_names, feature_types)