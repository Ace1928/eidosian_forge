import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
class TestStencilBase(unittest.TestCase):
    _numba_parallel_test_ = False

    def __init__(self, *args):
        self.cflags = Flags()
        self.cflags.nrt = True
        super(TestStencilBase, self).__init__(*args)

    def _compile_this(self, func, sig, flags):
        return compile_extra(registry.cpu_target.typing_context, registry.cpu_target.target_context, func, sig, None, flags, {})

    def compile_parallel(self, func, sig, **kws):
        flags = Flags()
        flags.nrt = True
        options = True if not kws else kws
        flags.auto_parallel = ParallelOptions(options)
        return self._compile_this(func, sig, flags)

    def compile_njit(self, func, sig):
        return self._compile_this(func, sig, flags=self.cflags)

    def compile_all(self, pyfunc, *args, **kwargs):
        sig = tuple([numba.typeof(x) for x in args])
        cpfunc = self.compile_parallel(pyfunc, sig)
        cfunc = self.compile_njit(pyfunc, sig)
        return (cfunc, cpfunc)

    def check(self, no_stencil_func, pyfunc, *args):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        expected = no_stencil_func(*args)
        py_output = pyfunc(*args)
        njit_output = cfunc.entry_point(*args)
        parfor_output = cpfunc.entry_point(*args)
        np.testing.assert_almost_equal(py_output, expected, decimal=3)
        np.testing.assert_almost_equal(njit_output, expected, decimal=3)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=3)
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())