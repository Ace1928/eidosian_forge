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
def _fail_test_with_unexpected_errors(tb, msg_format, cupy_error, numpy_error):
    msg = msg_format.format(cupy_error=_format_exception(cupy_error), numpy_error=_format_exception(numpy_error))
    raise AssertionError(msg).with_traceback(tb)