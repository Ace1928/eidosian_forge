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
@skip_if_typeguard
class TestCachingOverloadObjmode(TestCase):
    """Test caching of the use of overload implementations that use
    `with objmode`
    """
    _numba_parallel_test_ = False

    def setUp(self):
        warnings.simplefilter('error', errors.NumbaWarning)

    def tearDown(self):
        warnings.resetwarnings()

    def test_caching_overload_objmode(self):
        cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', cache_dir):

            def realwork(x):
                arr = np.arange(x) / x
                return np.linalg.norm(arr)

            def python_code(x):
                return realwork(x)

            @overload(with_objmode_cache_ov_example)
            def _ov_with_objmode_cache_ov_example(x):

                def impl(x):
                    with objmode(y='float64'):
                        y = python_code(x)
                    return y
                return impl

            @njit(cache=True)
            def testcase(x):
                return with_objmode_cache_ov_example(x)
            expect = realwork(123)
            got = testcase(123)
            self.assertEqual(got, expect)
            testcase_cached = njit(cache=True)(testcase.py_func)
            got = testcase_cached(123)
            self.assertEqual(got, expect)

    @classmethod
    def check_objmode_cache_ndarray(cls):

        def do_this(a, b):
            return np.sum(a + b)

        def do_something(a, b):
            return np.sum(a + b)

        @overload(do_something)
        def overload_do_something(a, b):

            def _do_something_impl(a, b):
                with objmode(y='float64'):
                    y = do_this(a, b)
                return y
            return _do_something_impl

        @njit(cache=True)
        def test_caching():
            a = np.arange(20)
            b = np.arange(20)
            return do_something(a, b)
        got = test_caching()
        expect = test_caching.py_func()
        if got != expect:
            raise AssertionError('incorrect result')
        return test_caching

    @classmethod
    def populate_objmode_cache_ndarray_check_cache(cls):
        cls.check_objmode_cache_ndarray()

    @classmethod
    def check_objmode_cache_ndarray_check_cache(cls):
        disp = cls.check_objmode_cache_ndarray()
        if len(disp.stats.cache_misses) != 0:
            raise AssertionError('unexpected cache miss')
        if len(disp.stats.cache_hits) <= 0:
            raise AssertionError('unexpected missing cache hit')

    def test_check_objmode_cache_ndarray(self):
        cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', cache_dir):
            run_in_new_process_in_cache_dir(self.populate_objmode_cache_ndarray_check_cache, cache_dir)
            res = run_in_new_process_in_cache_dir(self.check_objmode_cache_ndarray_check_cache, cache_dir)
        self.assertEqual(res['exitcode'], 0)