from enum import Enum
from typing import Dict, Union, List, TYPE_CHECKING
import warnings
import numpy as np
from ray.air.data_batch_type import DataBatchType
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.util.annotations import Deprecated, DeveloperAPI
def _unwrap_ndarray_object_type_if_needed(arr: np.ndarray) -> np.ndarray:
    """Unwrap an object-dtyped NumPy ndarray containing ndarray pointers into a single
    contiguous ndarray, if needed/possible.
    """
    if arr.dtype.type is np.object_:
        try:
            arr = np.array([np.asarray(v) for v in arr])
        except Exception:
            pass
    return arr