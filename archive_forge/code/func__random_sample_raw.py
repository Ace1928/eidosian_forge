import atexit
import binascii
import functools
import hashlib
import operator
import os
import time
import numpy
import warnings
from numpy.linalg import LinAlgError
import cupy
from cupy import _core
from cupy import cuda
from cupy.cuda import curand
from cupy.cuda import device
from cupy.random import _kernels
from cupy import _util
import cupyx
def _random_sample_raw(self, size, dtype):
    dtype = _check_and_get_dtype(dtype)
    out = cupy.empty(size, dtype=dtype)
    if dtype.char == 'f':
        func = curand.generateUniform
    else:
        func = curand.generateUniformDouble
    func(self._generator, out.data.ptr, out.size)
    return out