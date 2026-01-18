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
def _test_root_function(self, fn=pos_root):
    A = np.random.random(10)
    B = np.random.random(10) + 1.0
    C = np.random.random(10)
    arg_tys = [typeof(arg) for arg in (A, B, C)]
    control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys)
    control_cres = control_pipeline.compile_extra(fn)
    nb_fn_0 = control_cres.entry_point
    test_pipeline = RewritesTester.mk_pipeline(arg_tys)
    test_cres = test_pipeline.compile_extra(fn)
    nb_fn_1 = test_cres.entry_point
    np_result = fn(A, B, C)
    nb_result_0 = nb_fn_0(A, B, C)
    nb_result_1 = nb_fn_1(A, B, C)
    np.testing.assert_array_almost_equal(np_result, nb_result_0)
    np.testing.assert_array_almost_equal(nb_result_0, nb_result_1)
    return Namespace(locals())