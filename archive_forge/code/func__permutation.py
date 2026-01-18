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
def _permutation(self, num):
    """Returns a permuted range."""
    sample = cupy.empty((num,), dtype=numpy.int32)
    curand.generate(self._generator, sample.data.ptr, num)
    array = cupy.argsort(sample)
    return array