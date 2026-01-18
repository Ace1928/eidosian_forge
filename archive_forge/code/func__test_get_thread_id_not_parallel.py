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
def _test_get_thread_id_not_parallel(self):
    python_get_thread_id = get_thread_id()
    check_array_size = 8

    @njit(parallel=False)
    def par_false(size):
        njit_par_false_tid = get_thread_id()
        res = np.ones(size)
        for i in prange(size):
            res[i] = get_thread_id()
        return (njit_par_false_tid, res)

    @njit(parallel=True)
    def par_true(size):
        njit_par_true_tid = get_thread_id()
        res = np.ones(size)
        for i in range(size):
            res[i] = get_thread_id()
        return (njit_par_true_tid, res)
    self.assertEqual(python_get_thread_id, 0)
    njit_par_false_tid, njit_par_false_arr = par_false(check_array_size)
    self.assertEqual(njit_par_false_tid, 0)
    np.testing.assert_equal(njit_par_false_arr, 0)
    njit_par_true_tid, njit_par_true_arr = par_true(check_array_size)
    self.assertEqual(njit_par_true_tid, 0)
    np.testing.assert_equal(njit_par_true_arr, 0)