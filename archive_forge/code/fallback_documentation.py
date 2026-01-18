import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification

        Updates _cupy_array from _numpy_array.
        To be executed before calling cupy function.
        