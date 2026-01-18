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
def _test_nested_parallelism_1(self):
    if threading_layer() == 'workqueue':
        self.skipTest('workqueue is not threadsafe')
    mask = config.NUMBA_NUM_THREADS - 1
    N = config.NUMBA_NUM_THREADS
    M = 2 * config.NUMBA_NUM_THREADS

    @njit(parallel=True)
    def child_func(buf, fid):
        M, N = buf.shape
        for i in prange(N):
            buf[fid, i] = get_num_threads()

    def get_test(test_type):
        if test_type == 'njit':

            def test_func(nthreads, py_func=False):

                @njit(parallel=True)
                def _test_func(nthreads):
                    acc = 0
                    buf = np.zeros((M, N))
                    set_num_threads(nthreads)
                    for i in prange(M):
                        local_mask = 1 + i % mask
                        set_num_threads(local_mask)
                        if local_mask < N:
                            child_func(buf, local_mask)
                        acc += get_num_threads()
                    return (acc, buf)
                if py_func:
                    return _test_func.py_func(nthreads)
                else:
                    return _test_func(nthreads)
        elif test_type == 'guvectorize':

            def test_func(nthreads, py_func=False):

                def _test_func(acc, buf, local_mask):
                    set_num_threads(nthreads)
                    set_num_threads(local_mask[0])
                    if local_mask[0] < N:
                        child_func(buf, local_mask[0])
                    acc[0] += get_num_threads()
                buf = np.zeros((M, N), dtype=np.int64)
                acc = np.zeros((M, 1), dtype=np.int64)
                local_mask = (1 + np.arange(M) % mask).reshape((M, 1))
                sig = ['void(int64[:], int64[:, :], int64[:])']
                layout = '(p), (n, m), (p)'
                if not py_func:
                    _test_func = guvectorize(sig, layout, nopython=True, target='parallel')(_test_func)
                else:
                    _test_func = guvectorize(sig, layout, forceobj=True)(_test_func)
                _test_func(acc, buf, local_mask)
                return (acc, buf)
        return test_func
    for test_type in ['njit', 'guvectorize']:
        test_func = get_test(test_type)
        got_acc, got_arr = test_func(mask)
        exp_acc, exp_arr = test_func(mask, py_func=True)
        np.testing.assert_equal(exp_acc, got_acc)
        np.testing.assert_equal(exp_arr, got_arr)
        math_acc_exp = 1 + np.arange(M) % mask
        if test_type == 'guvectorize':
            math_acc = math_acc_exp.reshape((M, 1))
        else:
            math_acc = np.sum(math_acc_exp)
        np.testing.assert_equal(math_acc, got_acc)
        math_arr = np.zeros((M, N))
        for i in range(1, N):
            math_arr[i, :] = i
        np.testing.assert_equal(math_arr, got_arr)