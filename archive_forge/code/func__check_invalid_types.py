import numba
import numpy as np
import sys
import itertools
import gc
from numba import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np.random.generator_methods import _get_proper_func
from numba.np.random.generator_core import next_uint32, next_uint64, next_double
from numpy.random import MT19937, Generator
from numba.core.errors import TypingError
from numba.tests.support import run_in_new_process_caching, SerialMixin
def _check_invalid_types(self, dist_func, arg_list, valid_args, invalid_args):
    rng = np.random.default_rng()
    for idx, _arg in enumerate(arg_list):
        curr_args = valid_args.copy()
        curr_args[idx] = invalid_args[idx]
        curr_args = [rng] + curr_args
        nb_dist_func = numba.njit(dist_func)
        with self.assertRaises(TypingError) as raises:
            nb_dist_func(*curr_args)
        self.assertIn(f'Argument {_arg} is not one of the expected type(s):', str(raises.exception))