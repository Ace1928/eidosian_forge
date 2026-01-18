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
class TestLoopTypesInt(_LoopTypesTester):
    _ufuncs = supported_ufuncs[:]
    _ufuncs.remove(np.power)
    _ufuncs.remove(np.reciprocal)
    _ufuncs.remove(np.left_shift)
    _ufuncs.remove(np.right_shift)
    _ufuncs.remove(np.subtract)
    _ufuncs.remove(np.negative)
    _required_types = '?bBhHiIlLqQ'
    _skip_types = 'fdFDmMO' + _LoopTypesTester._skip_types