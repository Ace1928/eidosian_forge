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