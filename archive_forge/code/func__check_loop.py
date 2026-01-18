import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
def _check_loop(self, fn, ufunc, loop):
    letter_types = loop[:ufunc.nin] + loop[-ufunc.nout:]
    supported_types = getattr(self, '_supported_types', [])
    if supported_types and any((l not in supported_types for l in letter_types)):
        return
    skip_types = getattr(self, '_skip_types', [])
    if any((l in skip_types for l in letter_types)):
        return
    required_types = getattr(self, '_required_types', [])
    if required_types and (not any((l in letter_types for l in required_types))):
        return
    self._check_ufunc_with_dtypes(fn, ufunc, letter_types)