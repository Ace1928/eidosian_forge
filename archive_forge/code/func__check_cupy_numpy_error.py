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
def _check_cupy_numpy_error(cupy_error, numpy_error, accept_error=False):
    if isinstance(cupy_error, _skip_classes) and isinstance(numpy_error, _skip_classes):
        if cupy_error.__class__ is not numpy_error.__class__:
            raise AssertionError('Both numpy and cupy were skipped but with different exceptions.')
        if cupy_error.args != numpy_error.args:
            raise AssertionError('Both numpy and cupy were skipped but with different causes.')
        raise numpy_error
    if os.environ.get('CUPY_CI', '') != '' and cupy_error is not None:
        frame = traceback.extract_tb(cupy_error.__traceback__)[-1]
        filename = os.path.basename(frame.filename)
        if filename == 'test_helper.py':
            pass
        elif filename.startswith('test_'):
            _fail_test_with_unexpected_errors(cupy_error.__traceback__, 'Error was raised from test code.\n\n{cupy_error}', cupy_error, None)
    if accept_error is True:
        accept_error = Exception
    elif not accept_error:
        accept_error = ()
    if cupy_error is None and numpy_error is None:
        raise AssertionError('Both cupy and numpy are expected to raise errors, but not')
    elif cupy_error is None:
        _fail_test_with_unexpected_errors(numpy_error.__traceback__, 'Only numpy raises error\n\n{numpy_error}', None, numpy_error)
    elif numpy_error is None:
        _fail_test_with_unexpected_errors(cupy_error.__traceback__, 'Only cupy raises error\n\n{cupy_error}', cupy_error, None)
    elif not _check_numpy_cupy_error_compatible(cupy_error, numpy_error):
        _fail_test_with_unexpected_errors(cupy_error.__traceback__, 'Different types of errors occurred\n\ncupy\n{cupy_error}\n\nnumpy\n{numpy_error}\n', cupy_error, numpy_error)
    elif not (isinstance(cupy_error, accept_error) and isinstance(numpy_error, accept_error)):
        _fail_test_with_unexpected_errors(cupy_error.__traceback__, 'Both cupy and numpy raise exceptions\n\ncupy\n{cupy_error}\n\nnumpy\n{numpy_error}\n', cupy_error, numpy_error)