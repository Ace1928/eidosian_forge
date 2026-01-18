import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage
import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray
import unittest
@contextlib.contextmanager
def _assert_no_warnings_context(name=None):
    __tracebackhide__ = True
    with warnings.catch_warnings(record=True) as l:
        warnings.simplefilter('always')
        yield
        if len(l) > 0:
            name_str = f' when calling {name}' if name is not None else ''
            raise AssertionError(f'Got warnings{name_str}: {l}')