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
Returns a scalar or an array of integer values over ``[low, high)``.

        .. seealso::
            - :func:`cupy.random.randint` for full documentation
            - :meth:`numpy.random.RandomState.randint`
        