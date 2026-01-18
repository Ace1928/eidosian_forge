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
def _call_func_cupy(impl, args, kw, name, sp_name, scipy_name):
    assert isinstance(name, str)
    assert sp_name is None or isinstance(sp_name, str)
    assert scipy_name is None or isinstance(scipy_name, str)
    kw = kw.copy()
    if sp_name:
        kw[sp_name] = cupyx.scipy.sparse
    if scipy_name:
        kw[scipy_name] = cupyx.scipy
    kw[name] = cupy
    result, error = _call_func(impl, args, kw)
    return (result, error)