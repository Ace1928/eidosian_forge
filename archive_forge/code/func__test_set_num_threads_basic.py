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
def _test_set_num_threads_basic(self):
    max_threads = config.NUMBA_NUM_THREADS
    self.assertEqual(get_num_threads(), max_threads)
    set_num_threads(2)
    self.assertEqual(get_num_threads(), 2)
    set_num_threads(max_threads)
    self.assertEqual(get_num_threads(), max_threads)
    with self.assertRaises(ValueError):
        set_num_threads(0)
    with self.assertRaises(ValueError):
        set_num_threads(max_threads + 1)