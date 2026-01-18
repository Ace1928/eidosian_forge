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
def _curand_generate(self, num, dtype):
    sample = cupy.empty((num,), dtype=dtype)
    size32 = sample.view(dtype=numpy.uint32).size
    curand.generate(self._generator, sample.data.ptr, size32)
    return sample