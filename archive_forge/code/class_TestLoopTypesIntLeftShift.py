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
class TestLoopTypesIntLeftShift(_LoopTypesTester):
    _ufuncs = [np.left_shift]
    _required_types = 'bBhHiIlLqQ'
    _skip_types = 'fdFDmMO' + _LoopTypesTester._skip_types

    def _arg_for_type(self, a_letter_type, index=0):
        res = super(self.__class__, self)._arg_for_type(a_letter_type, index=index)
        if index == 1:
            bit_count = res.dtype.itemsize * 8
            res = np.clip(res, 0, bit_count - 1)
        return res