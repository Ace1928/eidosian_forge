import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _transform_dt_df(data: DataType, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes], meta: Optional[str]=None, meta_type: Optional[NumpyDType]=None) -> Tuple[np.ndarray, Optional[FeatureNames], Optional[FeatureTypes]]:
    """Validate feature names and types if data table"""
    _dt_type_mapper = {'bool': 'bool', 'int': 'int', 'real': 'float'}
    _dt_type_mapper2 = {'bool': 'i', 'int': 'int', 'real': 'float'}
    if meta and data.shape[1] > 1:
        raise ValueError('DataTable for meta info cannot have multiple columns')
    if meta:
        meta_type = 'float' if meta_type is None else meta_type
        data = data.to_numpy()[:, 0].astype(meta_type)
        return (data, None, None)
    data_types_names = tuple((lt.name for lt in data.ltypes))
    bad_fields = [data.names[i] for i, type_name in enumerate(data_types_names) if type_name not in _dt_type_mapper]
    if bad_fields:
        msg = 'DataFrame.types for data must be int, float or bool.\n                Did not expect the data types in fields '
        raise ValueError(msg + ', '.join(bad_fields))
    if feature_names is None and meta is None:
        feature_names = data.names
        if feature_types is not None:
            raise ValueError('DataTable has own feature types, cannot pass them in.')
        feature_types = np.vectorize(_dt_type_mapper2.get)(data_types_names).tolist()
    return (data, feature_names, feature_types)