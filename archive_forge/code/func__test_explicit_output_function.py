import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def _test_explicit_output_function(self, fn):
    """
        Test function having a (a, b, out) signature where *out* is
        an output array the function writes into.
        """
    A = np.arange(10, dtype=np.float64)
    B = A + 1
    arg_tys = (typeof(A),) * 3
    control_pipeline, control_cfunc, test_pipeline, test_cfunc = self._compile_function(fn, arg_tys)

    def run_func(fn):
        out = np.zeros_like(A)
        fn(A, B, out)
        return out
    expected = run_func(fn)
    self.assertPreciseEqual(expected, run_func(control_cfunc))
    self.assertPreciseEqual(expected, run_func(test_cfunc))
    return Namespace(locals())