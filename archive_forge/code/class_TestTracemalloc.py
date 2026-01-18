import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
class TestTracemalloc(unittest.TestCase):
    """
    Test NRT-allocated memory can be tracked by tracemalloc.
    """

    def measure_memory_diff(self, func):
        try:
            import tracemalloc
        except ImportError:
            self.skipTest('tracemalloc not available')
        tracemalloc.start()
        try:
            before = tracemalloc.take_snapshot()
            res = func()
            after = tracemalloc.take_snapshot()
            del res
            return after.compare_to(before, 'lineno')
        finally:
            tracemalloc.stop()

    def test_snapshot(self):
        N = 1000000
        dtype = np.int8

        @njit
        def alloc_nrt_memory():
            """
            Allocate and return a large array.
            """
            return np.empty(N, dtype)

        def keep_memory():
            return alloc_nrt_memory()

        def release_memory():
            alloc_nrt_memory()
        alloc_lineno = keep_memory.__code__.co_firstlineno + 1
        alloc_nrt_memory()
        diff = self.measure_memory_diff(keep_memory)
        stat = diff[0]
        self.assertGreaterEqual(stat.size, N)
        self.assertLess(stat.size, N * 1.015, msg='Unexpected allocation overhead encountered. May be due to difference in CPython builds or running under coverage')
        frame = stat.traceback[0]
        self.assertEqual(os.path.basename(frame.filename), 'test_nrt.py')
        self.assertEqual(frame.lineno, alloc_lineno)
        diff = self.measure_memory_diff(release_memory)
        stat = diff[0]
        self.assertLess(stat.size, N * 0.01)