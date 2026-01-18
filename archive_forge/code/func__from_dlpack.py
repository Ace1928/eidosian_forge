import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _from_dlpack(data: DataType, missing: FloatCompatible, nthread: int, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes]) -> DispatchedDataBackendReturnType:
    data = _transform_dlpack(data)
    return _from_cupy_array(data, missing, nthread, feature_names, feature_types)