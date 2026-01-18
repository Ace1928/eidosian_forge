import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
class ConstructorLikeBaseTest(object):

    def mutate_array(self, arr):
        try:
            arr.fill(42)
        except (TypeError, ValueError):
            fill_value = b'x' * arr.dtype.itemsize
            arr.fill(fill_value)

    def check_like(self, pyfunc, dtype):

        def check_arr(arr):
            expected = pyfunc(arr)
            ret = cfunc(arr)
            self.assertEqual(ret.size, expected.size)
            self.assertEqual(ret.dtype, expected.dtype)
            self.assertStridesEqual(ret, expected)
            self.check_result_value(ret, expected)
            self.mutate_array(ret)
            self.mutate_array(expected)
            np.testing.assert_equal(ret, expected)
        orig = np.linspace(0, 5, 6).astype(dtype)
        cfunc = nrtjit(pyfunc)
        for shape in (6, (2, 3), (1, 2, 3), (3, 1, 2), ()):
            if shape == ():
                arr = orig[-1:].reshape(())
            else:
                arr = orig.reshape(shape)
            check_arr(arr)
            if arr.ndim > 0:
                check_arr(arr[::2])
            arr.flags['WRITEABLE'] = False
            with self.assertRaises(ValueError):
                arr[0] = 1
            check_arr(arr)
        check_arr(orig[0])