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
def _test_set_num_threads_inside_guvectorize(self):

    @guvectorize(['void(int64[:])'], '(n)', nopython=True, target='parallel')
    def test_func(x):
        set_num_threads(x[0])
        x[:] = get_num_threads()
    x = np.zeros((5000000,), dtype=np.int64)
    mask = 2
    x[0] = mask
    test_func(x)
    np.testing.assert_equal(x, mask)