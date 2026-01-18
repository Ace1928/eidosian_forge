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
class BaseUFuncTest(MemoryLeakMixin):

    def setUp(self):
        super(BaseUFuncTest, self).setUp()
        self.inputs = [(np.uint32(0), types.uint32), (np.uint32(1), types.uint32), (np.int32(-1), types.int32), (np.int32(0), types.int32), (np.int32(1), types.int32), (np.uint64(0), types.uint64), (np.uint64(1), types.uint64), (np.int64(-1), types.int64), (np.int64(0), types.int64), (np.int64(1), types.int64), (np.float32(-0.5), types.float32), (np.float32(0.0), types.float32), (np.float32(0.5), types.float32), (np.float64(-0.5), types.float64), (np.float64(0.0), types.float64), (np.float64(0.5), types.float64), (np.array([0, 1], dtype='u4'), types.Array(types.uint32, 1, 'C')), (np.array([0, 1], dtype='u8'), types.Array(types.uint64, 1, 'C')), (np.array([-1, 0, 1], dtype='i4'), types.Array(types.int32, 1, 'C')), (np.array([-1, 0, 1], dtype='i8'), types.Array(types.int64, 1, 'C')), (np.array([-0.5, 0.0, 0.5], dtype='f4'), types.Array(types.float32, 1, 'C')), (np.array([-0.5, 0.0, 0.5], dtype='f8'), types.Array(types.float64, 1, 'C')), (np.array([0, 1], dtype=np.int8), types.Array(types.int8, 1, 'C')), (np.array([0, 1], dtype=np.int16), types.Array(types.int16, 1, 'C')), (np.array([0, 1], dtype=np.uint8), types.Array(types.uint8, 1, 'C')), (np.array([0, 1], dtype=np.uint16), types.Array(types.uint16, 1, 'C'))]

    @functools.lru_cache(maxsize=None)
    def _compile(self, pyfunc, args, nrt=False):
        return njit(args, _nrt=nrt, no_rewrites=True)(pyfunc)

    def _determine_output_type(self, input_type, int_output_type=None, float_output_type=None):
        ty = input_type
        if isinstance(ty, types.Array):
            ndim = ty.ndim
            ty = ty.dtype
        else:
            ndim = 1
        if ty in types.signed_domain:
            if int_output_type:
                output_type = types.Array(int_output_type, ndim, 'C')
            else:
                output_type = types.Array(ty, ndim, 'C')
        elif ty in types.unsigned_domain:
            if int_output_type:
                output_type = types.Array(int_output_type, ndim, 'C')
            else:
                output_type = types.Array(ty, ndim, 'C')
        elif float_output_type:
            output_type = types.Array(float_output_type, ndim, 'C')
        else:
            output_type = types.Array(ty, ndim, 'C')
        return output_type