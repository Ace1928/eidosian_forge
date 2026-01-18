import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
@classmethod
def _store_array_from_cupy(cls, array):
    return cls(_initial_array=array, _supports_cupy=True)