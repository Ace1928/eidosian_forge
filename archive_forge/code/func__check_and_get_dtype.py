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
def _check_and_get_dtype(dtype):
    dtype = numpy.dtype(dtype)
    if dtype.char not in ('f', 'd'):
        raise TypeError('cupy.random only supports float32 and float64')
    return dtype