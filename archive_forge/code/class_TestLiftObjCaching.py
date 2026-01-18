import copy
import warnings
import numpy as np
import numba
from numba.core.transforms import find_setupwiths, with_lifting
from numba.core.withcontexts import bypass_context, call_context, objmode_context
from numba.core.bytecode import FunctionIdentity, ByteCode
from numba.core.interpreter import Interpreter
from numba.core import errors
from numba.core.registry import cpu_target
from numba.core.compiler import compile_ir, DEFAULT_FLAGS
from numba import njit, typeof, objmode, types
from numba.core.extending import overload
from numba.tests.support import (MemoryLeak, TestCase, captured_stdout,
from numba.core.utils import PYVERSION
from numba.experimental import jitclass
import unittest
class TestLiftObjCaching(MemoryLeak, TestCase):

    def setUp(self):
        warnings.simplefilter('error', errors.NumbaWarning)

    def tearDown(self):
        warnings.resetwarnings()

    def check(self, py_func):
        first = njit(cache=True)(py_func)
        self.assertEqual(first(123), 12.3)
        second = njit(cache=True)(py_func)
        self.assertFalse(second._cache_hits)
        self.assertEqual(second(123), 12.3)
        self.assertTrue(second._cache_hits)

    def test_objmode_caching_basic(self):

        def pyfunc(x):
            with objmode(output='float64'):
                output = x / 10
            return output
        self.check(pyfunc)

    def test_objmode_caching_call_closure_bad(self):

        def other_pyfunc(x):
            return x / 10

        def pyfunc(x):
            with objmode(output='float64'):
                output = other_pyfunc(x)
            return output
        self.check(pyfunc)

    def test_objmode_caching_call_closure_good(self):
        self.check(case_objmode_cache)