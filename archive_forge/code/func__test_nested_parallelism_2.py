from __future__ import print_function, absolute_import, division
import sys
import os
import re
import multiprocessing
import unittest
import numpy as np
from numba import (njit, set_num_threads, get_num_threads, prange, config,
from numba.np.ufunc.parallel import get_thread_id
from numba.core.errors import TypingError
from numba.tests.support import TestCase, skip_parfors_unsupported, tag
from numba.tests.test_parallel_backend import TestInSubprocess
@skip_parfors_unsupported
@unittest.skipIf(config.NUMBA_NUM_THREADS < 2, 'Not enough CPU cores')
def _test_nested_parallelism_2(self):
    if threading_layer() == 'workqueue':
        self.skipTest('workqueue is not threadsafe')
    N = config.NUMBA_NUM_THREADS + 1
    M = 4 * config.NUMBA_NUM_THREADS + 1

    def get_impl(child_type, test_type):
        if child_type == 'parallel':
            child_dec = njit(parallel=True)
        elif child_type == 'njit':
            child_dec = njit(parallel=False)
        elif child_type == 'none':

            def child_dec(x):
                return x

        @child_dec
        def child(buf, fid):
            M, N = buf.shape
            set_num_threads(fid)
            for i in prange(N):
                buf[fid, i] = get_num_threads()
        if test_type in ['parallel', 'njit', 'none']:
            if test_type == 'parallel':
                test_dec = njit(parallel=True)
            elif test_type == 'njit':
                test_dec = njit(parallel=False)
            elif test_type == 'none':

                def test_dec(x):
                    return x

            @test_dec
            def test_func(nthreads):
                buf = np.zeros((M, N))
                set_num_threads(nthreads)
                for i in prange(M):
                    local_mask = 1 + i % mask
                    if local_mask < config.NUMBA_NUM_THREADS:
                        child(buf, local_mask)
                        assert get_num_threads() == local_mask
                return buf
        else:
            if test_type == 'guvectorize':
                test_dec = guvectorize(['int64[:,:], int64[:]'], '(n, m), (k)', nopython=True, target='parallel')
            elif test_type == 'guvectorize-obj':
                test_dec = guvectorize(['int64[:,:], int64[:]'], '(n, m), (k)', forceobj=True)

            def test_func(nthreads):

                @test_dec
                def _test_func(buf, local_mask):
                    set_num_threads(nthreads)
                    if local_mask[0] < config.NUMBA_NUM_THREADS:
                        child(buf, local_mask[0])
                        assert get_num_threads() == local_mask[0]
                buf = np.zeros((M, N), dtype=np.int64)
                local_mask = (1 + np.arange(M) % mask).reshape((M, 1))
                _test_func(buf, local_mask)
                return buf
        return test_func
    mask = config.NUMBA_NUM_THREADS - 1
    res_arrays = {}
    for test_type in ['parallel', 'njit', 'none', 'guvectorize', 'guvectorize-obj']:
        for child_type in ['parallel', 'njit', 'none']:
            if child_type == 'none' and test_type != 'none':
                continue
            set_num_threads(mask)
            res_arrays[test_type, child_type] = get_impl(child_type, test_type)(mask)
    py_arr = res_arrays['none', 'none']
    for arr in res_arrays.values():
        np.testing.assert_equal(arr, py_arr)
    math_arr = np.zeros((M, N))
    for i in range(1, config.NUMBA_NUM_THREADS):
        math_arr[i, :] = i
    np.testing.assert_equal(math_arr, py_arr)