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
def _resolve_tolerance(type_check, result, rtol, atol):

    def _resolve(dtype, tol):
        if isinstance(tol, dict):
            tol1 = tol.get(dtype.type)
            if tol1 is None:
                tol1 = tol.get('default')
                if tol1 is None:
                    raise TypeError('Can not find tolerance for {}'.format(dtype.type))
            return tol1
        else:
            return tol
    dtype = result.dtype
    rtol1 = _resolve(dtype, rtol)
    atol1 = _resolve(dtype, atol)
    return (rtol1, atol1)