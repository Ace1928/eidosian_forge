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
def _check_ufunc_with_dtypes(self, fn, ufunc, dtypes):
    dtypes_with_units = []
    for t in dtypes:
        if t in ('m', 'M'):
            t = t + '8[D]'
        dtypes_with_units.append(t)
    arg_dty = [np.dtype(t) for t in dtypes_with_units]
    arg_nbty = tuple([types.Array(from_dtype(t), 1, 'C') for t in arg_dty])
    cfunc = njit(arg_nbty)(fn)
    c_args = [self._arg_for_type(t, index=index).repeat(2) for index, t in enumerate(dtypes)]
    for arr in c_args:
        self.random.shuffle(arr)
    py_args = [a.copy() for a in c_args]
    cfunc(*c_args)
    fn(*py_args)
    for dtype, py_arg, c_arg in zip(arg_dty, py_args, c_args):
        py_arg, c_arg = self._fixup_results(dtype, py_arg, c_arg)
        typechar = c_arg.dtype.char
        ulps = self._ulps.get((ufunc.__name__, typechar), 1)
        prec = 'single' if typechar in 'fF' else 'exact'
        prec = 'double' if typechar in 'dD' else prec
        msg = '\n'.join(["ufunc '{0}' arrays differ ({1}):", 'args: {2}', 'expected {3}', 'got {4}'])
        msg = msg.format(ufunc.__name__, c_args, prec, py_arg, c_arg)
        self.assertPreciseEqual(py_arg, c_arg, prec=prec, msg=msg, ulps=ulps)