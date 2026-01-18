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
def _make_positive_masks(impl, args, kw, name, sp_name, scipy_name):
    ks = [k for k, v in kw.items() if v in _unsigned_dtypes]
    for k in ks:
        kw[k] = _signed_counterpart(kw[k])
    result, error = _call_func_cupy(impl, args, kw, name, sp_name, scipy_name)
    assert error is None
    if not isinstance(result, (tuple, list)):
        result = (result,)
    return [cupy.asnumpy(r) >= 0 for r in result]