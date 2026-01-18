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
def _compile_function(self, fn, arg_tys):
    """
        Compile the given function both without and with rewrites enabled.
        """
    control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys)
    cres_0 = control_pipeline.compile_extra(fn)
    control_cfunc = cres_0.entry_point
    test_pipeline = RewritesTester.mk_pipeline(arg_tys)
    cres_1 = test_pipeline.compile_extra(fn)
    test_cfunc = cres_1.entry_point
    return (control_pipeline, control_cfunc, test_pipeline, test_cfunc)