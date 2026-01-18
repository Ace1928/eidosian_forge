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
class TestLoopTypesFloat(_LoopTypesTester):
    _ufuncs = supported_ufuncs[:]
    if iswindows:
        _ufuncs.remove(np.signbit)
    _ufuncs.remove(np.floor_divide)
    _ufuncs.remove(np.remainder)
    _ufuncs.remove(np.divmod)
    _ufuncs.remove(np.mod)
    _required_types = 'fd'
    _skip_types = 'FDmMO' + _LoopTypesTester._skip_types