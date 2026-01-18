import atexit
import contextlib
import functools
import importlib
import inspect
import os
import os.path as op
import re
import shutil
import sys
import tempfile
import unittest
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from subprocess import STDOUT, CalledProcessError, TimeoutExpired, check_output
from unittest import TestCase
import joblib
import numpy as np
import scipy as sp
from numpy.testing import assert_allclose as np_assert_allclose
from numpy.testing import (
import sklearn
from sklearn.utils import (
from sklearn.utils._array_api import _check_array_api_dispatch
from sklearn.utils.fixes import VisibleDeprecationWarning, parse_version, sp_version
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
def _array_api_for_tests(array_namespace, device):
    try:
        if array_namespace == 'numpy.array_api':
            with ignore_warnings(category=UserWarning):
                array_mod = importlib.import_module(array_namespace)
        else:
            array_mod = importlib.import_module(array_namespace)
    except ModuleNotFoundError:
        raise SkipTest(f'{array_namespace} is not installed: not checking array_api input')
    try:
        import array_api_compat
    except ImportError:
        raise SkipTest('array_api_compat is not installed: not checking array_api input')
    xp = array_api_compat.get_namespace(array_mod.asarray(1))
    if array_namespace == 'torch' and device == 'cuda' and (not xp.backends.cuda.is_built()):
        raise SkipTest('PyTorch test requires cuda, which is not available')
    elif array_namespace == 'torch' and device == 'mps':
        if os.getenv('PYTORCH_ENABLE_MPS_FALLBACK') != '1':
            raise SkipTest('Skipping MPS device test because PYTORCH_ENABLE_MPS_FALLBACK is not set.')
        if not xp.backends.mps.is_built():
            raise SkipTest('MPS is not available because the current PyTorch install was not built with MPS enabled.')
    elif array_namespace in {'cupy', 'cupy.array_api'}:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() == 0:
            raise SkipTest('CuPy test requires cuda, which is not available')
    return xp