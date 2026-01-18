import ctypes
import ctypes.util
import os
import sys
import threading
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core import errors
from numba.tests.support import TestCase, tag
class TestGILRelease(TestCase):

    def make_test_array(self, n_members):
        return np.arange(n_members, dtype=np.int64)

    def run_in_threads(self, func, n_threads):
        threads = []
        func(self.make_test_array(1), np.arange(1, dtype=np.intp))
        arr = self.make_test_array(50)
        for i in range(n_threads):
            indices = np.arange(arr.size, dtype=np.intp)
            np.random.shuffle(indices)
            t = threading.Thread(target=func, args=(arr, indices))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return arr

    def check_gil_held(self, func):
        arr = self.run_in_threads(func, n_threads=4)
        distinct = set(arr)
        self.assertEqual(len(distinct), 1, distinct)

    def check_gil_released(self, func):
        for n_threads in (4, 12, 32):
            arr = self.run_in_threads(func, n_threads)
            distinct = set(arr)
            try:
                self.assertGreater(len(distinct), 1, distinct)
            except AssertionError as e:
                failure = e
            else:
                return
        raise failure

    def test_gil_held(self):
        """
        Test the GIL is held by default, by checking serialized runs
        produce deterministic results.
        """
        cfunc = jit(f_sig, nopython=True)(f)
        self.check_gil_held(cfunc)

    def test_gil_released(self):
        """
        Test releasing the GIL, by checking parallel runs produce
        unpredictable results.
        """
        cfunc = jit(f_sig, nopython=True, nogil=True)(f)
        self.check_gil_released(cfunc)

    def test_gil_released_inside_lifted_loop(self):
        """
        Test the GIL can by released by a lifted loop even though the
        surrounding code uses object mode.
        """
        cfunc = jit(f_sig, forceobj=True, nogil=True)(lifted_f)
        self.check_gil_released(cfunc)

    def test_gil_released_by_caller(self):
        """
        Releasing the GIL in the caller is sufficient to have it
        released in a callee.
        """
        compiled_f = jit(f_sig, nopython=True)(f)

        @jit(f_sig, nopython=True, nogil=True)
        def caller(a, i):
            compiled_f(a, i)
        self.check_gil_released(caller)

    def test_gil_released_by_caller_and_callee(self):
        """
        Same, but with both caller and callee asking to release the GIL.
        """
        compiled_f = jit(f_sig, nopython=True, nogil=True)(f)

        @jit(f_sig, nopython=True, nogil=True)
        def caller(a, i):
            compiled_f(a, i)
        self.check_gil_released(caller)

    def test_gil_ignored_by_callee(self):
        """
        When only the callee asks to release the GIL, it gets ignored.
        """
        compiled_f = jit(f_sig, nopython=True, nogil=True)(f)

        @jit(f_sig, nopython=True)
        def caller(a, i):
            compiled_f(a, i)
        self.check_gil_held(caller)

    def test_object_mode(self):
        """
        When the function is compiled in object mode, a warning is
        printed out.
        """
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always', errors.NumbaWarning)
            cfunc = jit(f_sig, forceobj=True, nogil=True)(object_f)
        self.assertTrue(any((w.category is errors.NumbaWarning and "Code running in object mode won't allow parallel execution" in str(w.message) for w in wlist)), wlist)
        self.run_in_threads(cfunc, 2)