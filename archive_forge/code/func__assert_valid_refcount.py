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
def _assert_valid_refcount(op):
    """
    Check that ufuncs don't mishandle refcount of object `1`.
    Used in a few regression tests.
    """
    if not HAS_REFCOUNT:
        return True
    import gc
    import numpy as np
    b = np.arange(100 * 100).reshape(100, 100)
    c = b
    i = 1
    gc.disable()
    try:
        rc = sys.getrefcount(i)
        for j in range(15):
            d = op(b, c)
        assert_(sys.getrefcount(i) >= rc)
    finally:
        gc.enable()
    del d