import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _from_pandas_df(data: DataFrame, enable_categorical: bool, missing: FloatCompatible, nthread: int, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes]) -> DispatchedDataBackendReturnType:
    data, feature_names, feature_types = _transform_pandas_df(data, enable_categorical, feature_names, feature_types)
    return _from_numpy_array(data, missing, nthread, feature_names, feature_types)