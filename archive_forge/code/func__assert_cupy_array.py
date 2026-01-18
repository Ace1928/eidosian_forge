import os
import numpy
from numpy import linalg
import cupy
import cupy._util
from cupy import _core
import cupyx
def _assert_cupy_array(*arrays):
    for a in arrays:
        if not isinstance(a, cupy._core.ndarray):
            raise linalg.LinAlgError('cupy.linalg only supports cupy.ndarray')