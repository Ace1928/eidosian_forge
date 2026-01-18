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
def _test_bitgen_func_parity(self, func_name, bitgen_func, seed=1):
    numba_rng_instance = np.random.default_rng(seed=seed)
    numpy_rng_instance = np.random.default_rng(seed=seed)
    numpy_func = getattr(numpy_rng_instance.bit_generator.ctypes, func_name)
    numpy_res = numpy_func(numpy_rng_instance.bit_generator.ctypes.state)
    numba_func = numba.njit(lambda x: bitgen_func(x.bit_generator))
    numba_res = numba_func(numba_rng_instance)
    self.assertPreciseEqual(numba_res, numpy_res)