import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _transform_cudf_df(data: DataType, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes], enable_categorical: bool) -> Tuple[ctypes.c_void_p, list, Optional[FeatureNames], Optional[FeatureTypes]]:
    try:
        from cudf.api.types import is_categorical_dtype
    except ImportError:
        from cudf.utils.dtypes import is_categorical_dtype
    if _is_cudf_ser(data):
        dtypes = [data.dtype]
    else:
        dtypes = data.dtypes
    if not all((dtype.name in _pandas_dtype_mapper or (is_categorical_dtype(dtype) and enable_categorical) for dtype in dtypes)):
        _invalid_dataframe_dtype(data)
    if feature_names is None:
        if _is_cudf_ser(data):
            feature_names = [data.name]
        elif lazy_isinstance(data.columns, 'cudf.core.multiindex', 'MultiIndex'):
            feature_names = [' '.join([str(x) for x in i]) for i in data.columns]
        elif lazy_isinstance(data.columns, 'cudf.core.index', 'RangeIndex') or lazy_isinstance(data.columns, 'cudf.core.index', 'Int64Index') or lazy_isinstance(data.columns, 'cudf.core.index', 'Int32Index'):
            feature_names = list(map(str, data.columns))
        else:
            feature_names = data.columns.format()
    if feature_types is None:
        feature_types = []
        for dtype in dtypes:
            if is_categorical_dtype(dtype) and enable_categorical:
                feature_types.append(CAT_T)
            else:
                feature_types.append(_pandas_dtype_mapper[dtype.name])
    cat_codes = []
    if _is_cudf_ser(data):
        if is_categorical_dtype(data.dtype) and enable_categorical:
            codes = data.cat.codes
            cat_codes.append(codes)
    else:
        for col in data:
            dtype = data[col].dtype
            if is_categorical_dtype(dtype) and enable_categorical:
                codes = data[col].cat.codes
                cat_codes.append(codes)
            elif is_categorical_dtype(dtype):
                raise ValueError(_ENABLE_CAT_ERR)
            else:
                cat_codes.append([])
    return (data, cat_codes, feature_names, feature_types)