import functools
import inspect
import os
import random
from typing import Tuple, Type
import traceback
import unittest
import warnings
import numpy
import cupy
from cupy.testing import _array
from cupy.testing import _parameterized
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def _check_tolerance_keys(rtol, atol):

    def _check(tol):
        if isinstance(tol, dict):
            for k in tol.keys():
                if type(k) is type:
                    continue
                if type(k) is str and k == 'default':
                    continue
                msg = "Keys of the tolerance dictionary need to be type objects as `numpy.float32` and `cupy.float32` or `'default'` string."
                raise TypeError(msg)
    _check(rtol)
    _check(atol)