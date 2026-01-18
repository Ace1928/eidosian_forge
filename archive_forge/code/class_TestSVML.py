import math
import numpy as np
import numbers
import re
import traceback
import multiprocessing as mp
import numba
from numba import njit, prange
from numba.core import config
from numba.tests.support import TestCase, tag, override_env_config
import unittest
@needs_svml
class TestSVML(TestCase):
    """ Tests SVML behaves as expected """
    _numba_parallel_test_ = False

    def compile(self, func, *args, **kwargs):
        assert not kwargs
        sig = tuple([numba.typeof(x) for x in args])
        std = njit(sig)(func)
        fast = njit(sig, fastmath=True)(func)
        return (std.overloads[sig], fast.overloads[sig])

    def copy_args(self, *args):
        if not args:
            return tuple()
        new_args = []
        for x in args:
            if isinstance(x, np.ndarray):
                new_args.append(x.copy('k'))
            elif isinstance(x, np.number):
                new_args.append(x.copy())
            elif isinstance(x, numbers.Number):
                new_args.append(x)
            else:
                raise ValueError('Unsupported argument type encountered')
        return tuple(new_args)

    def check_result(self, pyfunc, *args, **kwargs):
        jitstd, jitfast = self.compile(pyfunc, *args)
        py_expected = pyfunc(*self.copy_args(*args))
        jitstd_result = jitstd.entry_point(*self.copy_args(*args))
        jitfast_result = jitfast.entry_point(*self.copy_args(*args))
        np.testing.assert_almost_equal(jitstd_result, py_expected, **kwargs)
        np.testing.assert_almost_equal(jitfast_result, py_expected, **kwargs)

    def check_asm(self, pyfunc, *args, **kwargs):
        std_pattern = kwargs.pop('std_pattern', None)
        fast_pattern = kwargs.pop('fast_pattern', None)
        jitstd, jitfast = self.compile(pyfunc, *args)
        if std_pattern:
            self.check_svml_presence(jitstd, std_pattern)
        if fast_pattern:
            self.check_svml_presence(jitfast, fast_pattern)

    def check(self, pyfunc, *args, what='both', **kwargs):
        assert what in ('both', 'result', 'asm')
        if what == 'both' or what == 'result':
            self.check_result(pyfunc, *args, **kwargs)
        if what == 'both' or what == 'asm':
            self.check_asm(pyfunc, *args, **kwargs)

    def check_svml_presence(self, func, pattern):
        asm = func.library.get_asm_str()
        self.assertIn(pattern, asm)

    @TestCase.run_test_in_subprocess(envvars=_skylake_axv512_envvars)
    def test_scalar_context_asm(self):
        pat = '$_sin' if config.IS_OSX else '$sin'
        self.check(math_sin_scalar, 7.0, what='asm', std_pattern=pat)
        self.check(math_sin_scalar, 7.0, what='asm', fast_pattern=pat)

    def test_scalar_context_result(self):
        self.check(math_sin_scalar, 7.0, what='result')

    @TestCase.run_test_in_subprocess(envvars=_skylake_axv512_envvars)
    def test_svml_asm(self):
        std = '__svml_sin8_ha,'
        fast = '__svml_sin8,'
        self.check(math_sin_loop, 10, what='asm', std_pattern=std, fast_pattern=fast)

    def test_svml_result(self):
        self.check(math_sin_loop, 10, what='result')

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_DISABLE_INTEL_SVML': '1', **_skylake_axv512_envvars})
    def test_svml_disabled(self):

        def math_sin_loop(n):
            ret = np.empty(n, dtype=np.float64)
            for x in range(n):
                ret[x] = math.sin(np.float64(x))
            return ret
        sig = (numba.int32,)
        std = njit(sig)(math_sin_loop)
        fast = njit(sig, fastmath=True)(math_sin_loop)
        fns = (std.overloads[sig], fast.overloads[sig])
        for fn in fns:
            asm = fn.library.get_asm_str()
            self.assertNotIn('__svml_sin', asm)

    def test_svml_working_in_non_isolated_context(self):

        @njit(fastmath={'fast'}, error_model='numpy')
        def impl(n):
            x = np.empty(n * 8, dtype=np.float64)
            ret = np.empty_like(x)
            for i in range(ret.size):
                ret[i] += math.cosh(x[i])
            return ret
        impl(1)
        self.assertTrue('intel_svmlcc' in impl.inspect_llvm(impl.signatures[0]))