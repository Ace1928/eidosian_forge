import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
class TestJitOptionsNoNRT(TestCase):

    def check_error_no_nrt(self, func, *args, **kwargs):
        msg = 'Only accept returning of array passed into the function as argument'
        with self.assertRaises(errors.TypingError) as raises:
            func(*args, **kwargs)
        self.assertIn(msg, str(raises.exception))

    def no_nrt_overload_check(self, flag):

        def dummy():
            return np.arange(10)

        @overload(dummy, jit_options={'_nrt': flag})
        def ov_dummy():

            def dummy():
                return np.arange(10)
            return dummy

        @njit
        def foo():
            return dummy()
        if flag:
            self.assertPreciseEqual(foo(), np.arange(10))
        else:
            self.check_error_no_nrt(foo)

    def test_overload_no_nrt(self):
        self.no_nrt_overload_check(True)
        self.no_nrt_overload_check(False)

    def test_overload_method_no_nrt(self):

        @njit
        def udt(x):
            return x.method_jit_option_check_nrt()
        self.assertPreciseEqual(udt(mydummy), np.arange(10))

        @njit
        def udt(x):
            return x.method_jit_option_check_no_nrt()
        self.check_error_no_nrt(udt, mydummy)

    def test_overload_attribute_no_nrt(self):

        @njit
        def udt(x):
            return x.attr_jit_option_check_nrt
        self.assertPreciseEqual(udt(mydummy), np.arange(10))

        @njit
        def udt(x):
            return x.attr_jit_option_check_no_nrt
        self.check_error_no_nrt(udt, mydummy)