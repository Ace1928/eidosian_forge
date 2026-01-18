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
@skip_unsupported
class TestManyStencils(TestStencilBase):

    def __init__(self, *args, **kwargs):
        super(TestManyStencils, self).__init__(*args, **kwargs)

    def check_against_expected(self, pyfunc, expected, *args, **kwargs):
        """
        For a given kernel:

        The expected result is available from argument `expected`.

        The following results are then computed:
        * from a pure @stencil decoration of the kernel.
        * from the njit of a trivial wrapper function around the pure @stencil
          decorated function.
        * from the njit(parallel=True) of a trivial wrapper function around
           the pure @stencil decorated function.

        The results are then compared.
        """
        options = kwargs.get('options', dict())
        expected_exception = kwargs.get('expected_exception')
        DEBUG_OUTPUT = False
        should_fail = []
        should_not_fail = []

        @contextmanager
        def errorhandler(exty=None, usecase=None):
            try:
                yield
            except Exception as e:
                if exty is not None:
                    lexty = exty if hasattr(exty, '__iter__') else [exty]
                    found = False
                    for ex in lexty:
                        found |= isinstance(e, ex)
                    if not found:
                        raise
                else:
                    should_not_fail.append((usecase, '%s: %s' % (type(e), str(e))))
            else:
                if exty is not None:
                    should_fail.append(usecase)
        if isinstance(expected_exception, dict):
            stencil_ex = expected_exception['stencil']
            njit_ex = expected_exception['njit']
            parfor_ex = expected_exception['parfor']
        else:
            stencil_ex = expected_exception
            njit_ex = expected_exception
            parfor_ex = expected_exception
        stencil_args = {'func_or_mode': pyfunc}
        stencil_args.update(options)
        stencilfunc_output = None
        with errorhandler(stencil_ex, '@stencil'):
            stencil_func_impl = stencil(**stencil_args)
            stencilfunc_output = stencil_func_impl(*args)
        if len(args) == 1:

            def wrap_stencil(arg0):
                return stencil_func_impl(arg0)
        elif len(args) == 2:

            def wrap_stencil(arg0, arg1):
                return stencil_func_impl(arg0, arg1)
        elif len(args) == 3:

            def wrap_stencil(arg0, arg1, arg2):
                return stencil_func_impl(arg0, arg1, arg2)
        else:
            raise ValueError('Up to 3 arguments can be provided, found %s' % len(args))
        sig = tuple([numba.typeof(x) for x in args])
        njit_output = None
        with errorhandler(njit_ex, 'njit'):
            wrapped_cfunc = self.compile_njit(wrap_stencil, sig)
            njit_output = wrapped_cfunc.entry_point(*args)
        parfor_output = None
        with errorhandler(parfor_ex, 'parfors'):
            wrapped_cpfunc = self.compile_parallel(wrap_stencil, sig)
            parfor_output = wrapped_cpfunc.entry_point(*args)
        if DEBUG_OUTPUT:
            print('\n@stencil_output:\n', stencilfunc_output)
            print('\nnjit_output:\n', njit_output)
            print('\nparfor_output:\n', parfor_output)
        try:
            if not stencil_ex:
                np.testing.assert_almost_equal(stencilfunc_output, expected, decimal=1)
                self.assertEqual(expected.dtype, stencilfunc_output.dtype)
        except Exception as e:
            should_not_fail.append(('@stencil', '%s: %s' % (type(e), str(e))))
            print('@stencil failed: %s' % str(e))
        try:
            if not njit_ex:
                np.testing.assert_almost_equal(njit_output, expected, decimal=1)
                self.assertEqual(expected.dtype, njit_output.dtype)
        except Exception as e:
            should_not_fail.append(('njit', '%s: %s' % (type(e), str(e))))
            print('@njit failed: %s' % str(e))
        try:
            if not parfor_ex:
                np.testing.assert_almost_equal(parfor_output, expected, decimal=1)
                self.assertEqual(expected.dtype, parfor_output.dtype)
                try:
                    self.assertIn('@do_scheduling', wrapped_cpfunc.library.get_llvm_str())
                except AssertionError:
                    msg = 'Could not find `@do_scheduling` in LLVM IR'
                    raise AssertionError(msg)
        except Exception as e:
            should_not_fail.append(('parfors', '%s: %s' % (type(e), str(e))))
            print('@njit(parallel=True) failed: %s' % str(e))
        if DEBUG_OUTPUT:
            print('\n\n')
        if should_fail:
            msg = ['%s' % x for x in should_fail]
            raise RuntimeError('The following implementations should have raised an exception but did not:\n%s' % msg)
        if should_not_fail:
            impls = ['%s' % x[0] for x in should_not_fail]
            errs = ''.join(['%s: Message: %s\n\n' % x for x in should_not_fail])
            str1 = 'The following implementations should not have raised an exception but did:\n%s\n' % impls
            str2 = 'Errors were:\n\n%s' % errs
            raise RuntimeError(str1 + str2)

    def check_exceptions(self, pyfunc, *args, **kwargs):
        """
        For a given kernel:

        The expected result is computed from a pyStencil version of the
        stencil.

        The following results are then computed:
        * from a pure @stencil decoration of the kernel.
        * from the njit of a trivial wrapper function around the pure @stencil
          decorated function.
        * from the njit(parallel=True) of a trivial wrapper function around
           the pure @stencil decorated function.

        The results are then compared.
        """
        options = kwargs.get('options', dict())
        expected_exception = kwargs.get('expected_exception')
        should_fail = []
        should_not_fail = []

        @contextmanager
        def errorhandler(exty=None, usecase=None):
            try:
                yield
            except Exception as e:
                if exty is not None:
                    lexty = exty if hasattr(exty, '__iter__') else [exty]
                    found = False
                    for ex in lexty:
                        found |= isinstance(e, ex)
                    if not found:
                        raise
                else:
                    should_not_fail.append((usecase, '%s: %s' % (type(e), str(e))))
            else:
                if exty is not None:
                    should_fail.append(usecase)
        if isinstance(expected_exception, dict):
            stencil_ex = expected_exception['stencil']
            njit_ex = expected_exception['njit']
            parfor_ex = expected_exception['parfor']
        else:
            stencil_ex = expected_exception
            njit_ex = expected_exception
            parfor_ex = expected_exception
        stencil_args = {'func_or_mode': pyfunc}
        stencil_args.update(options)
        with errorhandler(stencil_ex, '@stencil'):
            stencil_func_impl = stencil(**stencil_args)
            stencil_func_impl(*args)
        if len(args) == 1:

            def wrap_stencil(arg0):
                return stencil_func_impl(arg0)
        elif len(args) == 2:

            def wrap_stencil(arg0, arg1):
                return stencil_func_impl(arg0, arg1)
        elif len(args) == 3:

            def wrap_stencil(arg0, arg1, arg2):
                return stencil_func_impl(arg0, arg1, arg2)
        else:
            raise ValueError('Up to 3 arguments can be provided, found %s' % len(args))
        sig = tuple([numba.typeof(x) for x in args])
        with errorhandler(njit_ex, 'njit'):
            wrapped_cfunc = self.compile_njit(wrap_stencil, sig)
            wrapped_cfunc.entry_point(*args)
        with errorhandler(parfor_ex, 'parfors'):
            wrapped_cpfunc = self.compile_parallel(wrap_stencil, sig)
            wrapped_cpfunc.entry_point(*args)
        if should_fail:
            msg = ['%s' % x for x in should_fail]
            raise RuntimeError('The following implementations should have raised an exception but did not:\n%s' % msg)
        if should_not_fail:
            impls = ['%s' % x[0] for x in should_not_fail]
            errs = ''.join(['%s: Message: %s\n\n' % x for x in should_not_fail])
            str1 = 'The following implementations should not have raised an exception but did:\n%s\n' % impls
            str2 = 'Errors were:\n\n%s' % errs
            raise RuntimeError(str1 + str2)

    def exception_dict(self, **kwargs):
        d = dict()
        d['pyStencil'] = None
        d['stencil'] = None
        d['njit'] = None
        d['parfor'] = None
        for k, v in kwargs.items():
            d[k] = v
        return d

    def check_stencil_arrays(self, *args, **kwargs):
        neighborhood = kwargs.get('neighborhood')
        init_shape = args[0].shape
        if neighborhood is not None:
            if len(init_shape) != len(neighborhood):
                raise ValueError('Invalid neighborhood supplied')
        for x in args[1:]:
            if hasattr(x, 'shape'):
                if init_shape != x.shape:
                    raise ValueError('Input stencil arrays do not commute')

    def test_basic00(self):
        """rel index"""

        def kernel(a):
            return a[0, 0]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 0]
            return __b0
        a = np.arange(12).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic01(self):
        """rel index add const"""

        def kernel(a):
            return a[0, 1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic02(self):
        """rel index add const"""

        def kernel(a):
            return a[0, -1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(1, a.shape[1]):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + -1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic03(self):
        """rel index add const"""

        def kernel(a):
            return a[1, 0]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 1, __b + 0]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic04(self):
        """rel index add const"""

        def kernel(a):
            return a[-1, 0]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(1, a.shape[0]):
                    __b0[__a, __b] = a[__a + -1, __b + 0]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic05(self):
        """rel index add const"""

        def kernel(a):
            return a[-1, 1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(1, a.shape[0]):
                    __b0[__a, __b] = a[__a + -1, __b + 1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic06(self):
        """rel index add const"""

        def kernel(a):
            return a[1, -1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(1, a.shape[1]):
                for __a in range(0, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 1, __b + -1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic07(self):
        """rel index add const"""

        def kernel(a):
            return a[1, 1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 1, __b + 1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic08(self):
        """rel index add const"""

        def kernel(a):
            return a[-1, -1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(1, a.shape[1]):
                for __a in range(1, a.shape[0]):
                    __b0[__a, __b] = a[__a + -1, __b + -1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic09(self):
        """rel index add const"""

        def kernel(a):
            return a[-2, 2]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 2):
                for __a in range(2, a.shape[0]):
                    __b0[__a, __b] = a[__a + -2, __b + 2]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic10(self):
        """rel index add const"""

        def kernel(a):
            return a[0, 0] + a[1, 0]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + a[__a + 1, __b + 0]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic11(self):
        """rel index add const"""

        def kernel(a):
            return a[-1, 0] + a[1, 0]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(1, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + -1, __b + 0] + a[__a + 1, __b + 0]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic12(self):
        """rel index add const"""

        def kernel(a):
            return a[-1, 1] + a[1, -1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(1, a.shape[1] - 1):
                for __a in range(1, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + -1, __b + 1] + a[__a + 1, __b + -1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic13(self):
        """rel index add const"""

        def kernel(a):
            return a[-1, -1] + a[1, 1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(1, a.shape[1] - 1):
                for __a in range(1, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + -1, __b + -1] + a[__a + 1, __b + 1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic14(self):
        """rel index add domain change const"""

        def kernel(a):
            return a[0, 0] + 1j

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + 1j
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic14b(self):
        """rel index add domain change const"""

        def kernel(a):
            t = 1j
            return a[0, 0] + t

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0]):
                    t = 1j
                    __b0[__a, __b] = a[__a + 0, __b + 0] + t
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic15(self):
        """two rel index, add const"""

        def kernel(a):
            return a[0, 0] + a[1, 0] + 1.0

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + a[__a + 1, __b + 0] + 1.0
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic17(self):
        """two rel index boundary test, add const"""

        def kernel(a):
            return a[0, 0] + a[2, 0] + 1.0

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0] - 2):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + a[__a + 2, __b + 0] + 1.0
            return __b0
        a = np.arange(12).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic18(self):
        """two rel index boundary test, add const"""

        def kernel(a):
            return a[0, 0] + a[-2, 0] + 1.0

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(2, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + a[__a + -2, __b + 0] + 1.0
            return __b0
        a = np.arange(12).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic19(self):
        """two rel index boundary test, add const"""

        def kernel(a):
            return a[0, 0] + a[0, 3] + 1.0

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 3):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + a[__a + 0, __b + 3] + 1.0
            return __b0
        a = np.arange(12).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic20(self):
        """two rel index boundary test, add const"""

        def kernel(a):
            return a[0, 0] + a[0, -3] + 1.0

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(3, a.shape[1]):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + a[__a + 0, __b + -3] + 1.0
            return __b0
        a = np.arange(12).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic21(self):
        """same rel, add const"""

        def kernel(a):
            return a[0, 0] + a[0, 0] + 1.0

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + a[__a + 0, __b + 0] + 1.0
            return __b0
        a = np.arange(12).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic22(self):
        """rel idx const expr folding, add const"""

        def kernel(a):
            return a[1 + 0, 0] + a[0, 0] + 1.0

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 1, __b + 0] + a[__a + 0, __b + 0] + 1.0
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic23(self):
        """rel idx, work in body"""

        def kernel(a):
            x = np.sin(10 + a[2, 1])
            return a[1 + 0, 0] + a[0, 0] + x

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0] - 2):
                    x = np.sin(10 + a[__a + 2, __b + 1])
                    __b0[__a, __b] = a[__a + 1, __b + 0] + a[__a + 0, __b + 0] + x
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic23a(self):
        """rel idx, dead code should not impact rel idx"""

        def kernel(a):
            x = np.sin(10 + a[2, 1])
            return a[1 + 0, 0] + a[0, 0]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0] - 2):
                    x = np.sin(10 + a[__a + 2, __b + 1])
                    __b0[__a, __b] = a[__a + 1, __b + 0] + a[__a + 0, __b + 0]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic24(self):
        """1d idx on 2d arr"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0] + 1.0
        self.check_exceptions(kernel, a, expected_exception=[TypingError])

    def test_basic25(self):
        """no idx on 2d arr"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return 1.0
        self.check_exceptions(kernel, a, expected_exception=[ValueError, NumbaValueError])

    def test_basic26(self):
        """3d arr"""

        def kernel(a):
            return a[0, 0, 0] - a[0, 1, 0] + 1.0

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __c in range(0, a.shape[2]):
                for __b in range(0, a.shape[1] - 1):
                    for __a in range(0, a.shape[0]):
                        __b0[__a, __b, __c] = a[__a + 0, __b + 0, __c + 0] - a[__a + 0, __b + 1, __c + 0] + 1.0
            return __b0
        a = np.arange(64).reshape(4, 8, 2)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic27(self):
        """4d arr"""

        def kernel(a):
            return a[0, 0, 0, 0] - a[0, 1, 0, -1] + 1.0

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __d in range(1, a.shape[3]):
                for __c in range(0, a.shape[2]):
                    for __b in range(0, a.shape[1] - 1):
                        for __a in range(0, a.shape[0]):
                            __b0[__a, __b, __c, __d] = a[__a + 0, __b + 0, __c + 0, __d + 0] - a[__a + 0, __b + 1, __c + 0, __d + -1] + 1.0
            return __b0
        a = np.arange(128).reshape(4, 8, 2, 2)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic28(self):
        """type widen """

        def kernel(a):
            return a[0, 0] + np.float64(10.0)

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + np.float64(10.0)
            return __b0
        a = np.arange(12).reshape(3, 4).astype(np.float32)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic29(self):
        """const index from func """
        a = np.arange(12.0).reshape(3, 4)

        def kernel(a):
            return a[0, int(np.cos(0))]
        self.check_exceptions(kernel, a, expected_exception=[ValueError, NumbaValueError, LoweringError])

    def test_basic30(self):
        """signed zeros"""

        def kernel(a):
            return a[-0, -0]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + -0, __b + -0]
            return __b0
        a = np.arange(12).reshape(3, 4).astype(np.float32)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic31(self):
        """does a const propagate? 2D"""

        def kernel(a):
            t = 1
            return a[t, 0]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0] - 1):
                    t = 1
                    __b0[__a, __b] = a[__a + t, __b + 0]
            return __b0
        a = np.arange(12).reshape(3, 4).astype(np.float32)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    @unittest.skip('constant folding not implemented')
    def test_basic31b(self):
        """does a const propagate?"""
        a = np.arange(12.0).reshape(3, 4)

        def kernel(a):
            s = 1
            t = 1 - s
            return a[t, 0]

    def test_basic31c(self):
        """does a const propagate? 1D"""

        def kernel(a):
            t = 1
            return a[t]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __a in range(0, a.shape[0] - 1):
                t = 1
                __b0[__a,] = a[__a + t]
            return __b0
        a = np.arange(12.0)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic32(self):
        """typed int index"""
        a = np.arange(12.0).reshape(3, 4)

        def kernel(a):
            return a[np.int8(1), 0]
        self.check_exceptions(kernel, a, expected_exception=[ValueError, NumbaValueError, LoweringError])

    def test_basic33(self):
        """add 0d array"""

        def kernel(a):
            return a[0, 0] + np.array(1)

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + np.array(1)
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic34(self):
        """More complex rel index with dependency on addition rel index"""

        def kernel(a):
            g = 4.0 + a[0, 1]
            return g + (a[0, 1] + a[1, 0] + a[0, -1] + np.sin(a[-2, 0]))

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(1, a.shape[1] - 1):
                for __a in range(2, a.shape[0] - 1):
                    g = 4.0 + a[__a + 0, __b + 1]
                    __b0[__a, __b] = g + (a[__a + 0, __b + 1] + a[__a + 1, __b + 0] + a[__a + 0, __b + -1] + np.sin(a[__a + -2, __b + 0]))
            return __b0
        a = np.arange(144).reshape(12, 12)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic35(self):
        """simple cval where cval is int but castable to dtype of float"""

        def kernel(a):
            return a[0, 1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 5, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a, options={'cval': 5})

    def test_basic36(self):
        """more complex with cval"""

        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 5.0, dtype=type(__retdtype))
            for __b in range(1, a.shape[1] - 1):
                for __a in range(0, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 0, __b + 1] + a[__a + 0, __b + -1] + a[__a + 1, __b + -1] + a[__a + 1, __b + -1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a, options={'cval': 5})

    def test_basic37(self):
        """cval is expr"""

        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 68.0, dtype=type(__retdtype))
            for __b in range(1, a.shape[1] - 1):
                for __a in range(0, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 0, __b + 1] + a[__a + 0, __b + -1] + a[__a + 1, __b + -1] + a[__a + 1, __b + -1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a, options={'cval': 5 + 63.0})

    def test_basic38(self):
        """cval is complex"""

        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
        a = np.arange(12.0).reshape(3, 4)
        ex = self.exception_dict(stencil=NumbaValueError, parfor=ValueError, njit=NumbaValueError)
        self.check_exceptions(kernel, a, options={'cval': 1j}, expected_exception=ex)

    def test_basic39(self):
        """cval is func expr"""

        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
        cval = np.sin(3.0) + np.cos(2)

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, cval, dtype=type(__retdtype))
            for __b in range(1, a.shape[1] - 1):
                for __a in range(0, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 0, __b + 1] + a[__a + 0, __b + -1] + a[__a + 1, __b + -1] + a[__a + 1, __b + -1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a, options={'cval': cval})

    def test_basic40(self):
        """2 args!"""

        def kernel(a, b):
            return a[0, 1] + b[0, -2]

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, b, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(2, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 1] + b[__a + 0, __b + -2]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, b, None)
        self.check_against_expected(kernel, expected, a, b)

    def test_basic41(self):
        """2 args! rel arrays wildly not same size!"""

        def kernel(a, b):
            return a[0, 1] + b[0, -2]
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(1.0).reshape(1, 1)
        self.check_exceptions(kernel, a, b, expected_exception=[ValueError, AssertionError])

    def test_basic42(self):
        """2 args! rel arrays very close in size"""

        def kernel(a, b):
            return a[0, 1] + b[0, -2]
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(9.0).reshape(3, 3)
        self.check_exceptions(kernel, a, b, expected_exception=[ValueError, AssertionError])

    def test_basic43(self):
        """2 args more complexity"""

        def kernel(a, b):
            return a[0, 1] + a[1, 2] + b[-2, 0] + b[0, -1]

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, b, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(1, a.shape[1] - 2):
                for __a in range(2, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 0, __b + 1] + a[__a + 1, __b + 2] + b[__a + -2, __b + 0] + b[__a + 0, __b + -1]
            return __b0
        a = np.arange(30.0).reshape(5, 6)
        b = np.arange(30.0).reshape(5, 6)
        expected = __kernel(a, b, None)
        self.check_against_expected(kernel, expected, a, b)

    def test_basic44(self):
        """2 args, has assignment before use"""

        def kernel(a, b):
            a[0, 1] = 12
            return a[0, 1]
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        self.check_exceptions(kernel, a, b, expected_exception=[ValueError, LoweringError])

    def test_basic45(self):
        """2 args, has assignment and then cross dependency"""

        def kernel(a, b):
            a[0, 1] = 12
            return a[0, 1] + a[1, 0]
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        self.check_exceptions(kernel, a, b, expected_exception=[ValueError, LoweringError])

    def test_basic46(self):
        """2 args, has cross relidx assignment"""

        def kernel(a, b):
            a[0, 1] = b[1, 2]
            return a[0, 1] + a[1, 0]
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        self.check_exceptions(kernel, a, b, expected_exception=[ValueError, LoweringError])

    def test_basic47(self):
        """3 args"""

        def kernel(a, b, c):
            return a[0, 1] + b[1, 0] + c[-1, 0]

        def __kernel(a, b, c, neighborhood):
            self.check_stencil_arrays(a, b, c, neighborhood=neighborhood)
            __retdtype = kernel(a, b, c)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(1, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 0, __b + 1] + b[__a + 1, __b + 0] + c[__a + -1, __b + 0]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        c = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, b, c, None)
        self.check_against_expected(kernel, expected, a, b, c)

    def test_basic48(self):
        """2 args, has assignment before use via memory alias"""

        def kernel(a):
            c = a.T
            c[:, :] = 10
            return a[0, 1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    c = a.T
                    c[:, :] = 10
                    __b0[__a, __b] = a[__a + 0, __b + 1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic49(self):
        """2 args, standard_indexing on second"""

        def kernel(a, b):
            return a[0, 1] + b[0, 3]

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 1] + b[0, 3]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, b, None)
        self.check_against_expected(kernel, expected, a, b, options={'standard_indexing': 'b'})

    @unittest.skip('dynamic range checking not implemented')
    def test_basic50(self):
        """2 args, standard_indexing OOB"""

        def kernel(a, b):
            return a[0, 1] + b[0, 15]

    def test_basic51(self):
        """2 args, standard_indexing, no relidx"""

        def kernel(a, b):
            return a[0, 1] + b[0, 2]
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        self.check_exceptions(kernel, a, b, options={'standard_indexing': ['a', 'b']}, expected_exception=[ValueError, NumbaValueError])

    def test_basic52(self):
        """3 args, standard_indexing on middle arg """

        def kernel(a, b, c):
            return a[0, 1] + b[0, 1] + c[1, 2]

        def __kernel(a, b, c, neighborhood):
            self.check_stencil_arrays(a, c, neighborhood=neighborhood)
            __retdtype = kernel(a, b, c)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 2):
                for __a in range(0, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + 0, __b + 1] + b[0, 1] + c[__a + 1, __b + 2]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(4.0).reshape(2, 2)
        c = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, b, c, None)
        self.check_against_expected(kernel, expected, a, b, c, options={'standard_indexing': 'b'})

    def test_basic53(self):
        """2 args, standard_indexing on variable that does not exist"""

        def kernel(a, b):
            return a[0, 1] + b[0, 2]
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        ex = self.exception_dict(stencil=Exception, parfor=ValueError, njit=Exception)
        self.check_exceptions(kernel, a, b, options={'standard_indexing': 'c'}, expected_exception=ex)

    def test_basic54(self):
        """2 args, standard_indexing, index from var"""

        def kernel(a, b):
            t = 2
            return a[0, 1] + b[0, t]

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    t = 2
                    __b0[__a, __b] = a[__a + 0, __b + 1] + b[0, t]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, b, None)
        self.check_against_expected(kernel, expected, a, b, options={'standard_indexing': 'b'})

    def test_basic55(self):
        """2 args, standard_indexing, index from more complex var"""

        def kernel(a, b):
            s = 1
            t = 2 - s
            return a[0, 1] + b[0, t]

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    s = 1
                    t = 2 - s
                    __b0[__a, __b] = a[__a + 0, __b + 1] + b[0, t]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, b, None)
        self.check_against_expected(kernel, expected, a, b, options={'standard_indexing': 'b'})

    def test_basic56(self):
        """2 args, standard_indexing, added complexity """

        def kernel(a, b):
            s = 1
            acc = 0
            for k in b[0, :]:
                acc += k
            t = 2 - s - 1
            return a[0, 1] + b[0, t] + acc

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    s = 1
                    acc = 0
                    for k in b[0, :]:
                        acc += k
                    t = 2 - s - 1
                    __b0[__a, __b] = a[__a + 0, __b + 1] + b[0, t] + acc
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, b, None)
        self.check_against_expected(kernel, expected, a, b, options={'standard_indexing': 'b'})

    def test_basic57(self):
        """2 args, standard_indexing, split index operation """

        def kernel(a, b):
            c = b[0]
            return a[0, 1] + c[1]

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    c = b[0]
                    __b0[__a, __b] = a[__a + 0, __b + 1] + c[1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, b, None)
        self.check_against_expected(kernel, expected, a, b, options={'standard_indexing': 'b'})

    def test_basic58(self):
        """2 args, standard_indexing, split index with broadcast mutation """

        def kernel(a, b):
            c = b[0] + 1
            return a[0, 1] + c[1]

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    c = b[0] + 1
                    __b0[__a, __b] = a[__a + 0, __b + 1] + c[1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, b, None)
        self.check_against_expected(kernel, expected, a, b, options={'standard_indexing': 'b'})

    def test_basic59(self):
        """3 args, mix of array, relative and standard indexing and const"""

        def kernel(a, b, c):
            return a[0, 1] + b[1, 1] + c

        def __kernel(a, b, c, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a, b, c)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 1] + b[1, 1] + c
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        c = 10
        expected = __kernel(a, b, c, None)
        self.check_against_expected(kernel, expected, a, b, c, options={'standard_indexing': ['b', 'c']})

    def test_basic60(self):
        """3 args, mix of array, relative and standard indexing,
        tuple pass through"""

        def kernel(a, b, c):
            return a[0, 1] + b[1, 1] + c[0]
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        c = (10,)
        ex = self.exception_dict(parfor=ValueError)
        self.check_exceptions(kernel, a, b, c, options={'standard_indexing': ['b', 'c']}, expected_exception=ex)

    def test_basic61(self):
        """2 args, standard_indexing on first"""

        def kernel(a, b):
            return a[0, 1] + b[1, 1]
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        self.check_exceptions(kernel, a, b, options={'standard_indexing': 'a'}, expected_exception=Exception)

    def test_basic62(self):
        """2 args, standard_indexing and cval"""

        def kernel(a, b):
            return a[0, 1] + b[1, 1]

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 10.0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 1] + b[1, 1]
            return __b0
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12.0).reshape(3, 4)
        expected = __kernel(a, b, None)
        self.check_against_expected(kernel, expected, a, b, options={'standard_indexing': 'b', 'cval': 10.0})

    def test_basic63(self):
        """2 args, standard_indexing applied to relative, should fail,
        non-const idx"""

        def kernel(a, b):
            return a[0, b[0, 1]]
        a = np.arange(12.0).reshape(3, 4)
        b = np.arange(12).reshape(3, 4)
        ex = self.exception_dict(stencil=NumbaValueError, parfor=ValueError, njit=NumbaValueError)
        self.check_exceptions(kernel, a, b, options={'standard_indexing': 'b'}, expected_exception=ex)

    def test_basic64(self):
        """1 arg that uses standard_indexing"""

        def kernel(a):
            return a[0, 0]
        a = np.arange(12.0).reshape(3, 4)
        self.check_exceptions(kernel, a, options={'standard_indexing': 'a'}, expected_exception=[ValueError, NumbaValueError])

    def test_basic65(self):
        """basic induced neighborhood test"""

        def kernel(a):
            cumul = 0
            for i in range(-29, 1):
                cumul += a[i]
            return cumul / 30

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __an in range(29, a.shape[0]):
                cumul = 0
                for i in range(-29, 1):
                    cumul += a[__an + i]
                __b0[__an,] = cumul / 30
            return __b0
        a = np.arange(60.0)
        nh = ((-29, 0),)
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic66(self):
        """basic const neighborhood test"""

        def kernel(a):
            cumul = 0
            for i in range(-29, 1):
                cumul += a[0]
            return cumul / 30

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __an in range(29, a.shape[0]):
                cumul = 0
                for i in range(-29, 1):
                    cumul += a[__an + 0]
                __b0[__an,] = cumul / 30
            return __b0
        a = np.arange(60.0)
        nh = ((-29, 0),)
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic67(self):
        """basic 2d induced neighborhood test"""

        def kernel(a):
            cumul = 0
            for i in range(-5, 1):
                for j in range(-10, 1):
                    cumul += a[i, j]
            return cumul / (10 * 5)

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __bn in range(10, a.shape[1]):
                for __an in range(5, a.shape[0]):
                    cumul = 0
                    for i in range(-5, 1):
                        for j in range(-10, 1):
                            cumul += a[__an + i, __bn + j]
                    __b0[__an, __bn] = cumul / 50
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        nh = ((-5, 0), (-10, 0))
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic67b(self):
        """basic 2d induced 1D neighborhood"""

        def kernel(a):
            cumul = 0
            for j in range(-10, 1):
                cumul += a[0, j]
            return cumul / (10 * 5)
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        self.check_exceptions(kernel, a, options={'neighborhood': ((-10, 0),)}, expected_exception=[TypingError, ValueError])

    def test_basic68(self):
        """basic 2d one induced, one cost neighborhood test"""

        def kernel(a):
            cumul = 0
            for i in range(-5, 1):
                for j in range(-10, 1):
                    cumul += a[i, 0]
            return cumul / (10 * 5)

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __bn in range(10, a.shape[1]):
                for __an in range(5, a.shape[0]):
                    cumul = 0
                    for i in range(-5, 1):
                        for j in range(-10, 1):
                            cumul += a[__an + i, __bn + 0]
                    __b0[__an, __bn] = cumul / 50
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        nh = ((-5, 0), (-10, 0))
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic69(self):
        """basic 2d two cost neighborhood test"""

        def kernel(a):
            cumul = 0
            for i in range(-5, 1):
                for j in range(-10, 1):
                    cumul += a[0, 0]
            return cumul / (10 * 5)

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __bn in range(10, a.shape[1]):
                for __an in range(5, a.shape[0]):
                    cumul = 0
                    for i in range(-5, 1):
                        for j in range(-10, 1):
                            cumul += a[__an + 0, __bn + 0]
                    __b0[__an, __bn] = cumul / 50
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        nh = ((-5, 0), (-10, 0))
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic70(self):
        """neighborhood adding complexity"""

        def kernel(a):
            cumul = 0
            zz = 12.0
            for i in range(-5, 1):
                t = zz + i
                for j in range(-10, 1):
                    cumul += a[i, j] + t
            return cumul / (10 * 5)

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __bn in range(10, a.shape[1]):
                for __an in range(5, a.shape[0]):
                    cumul = 0
                    zz = 12.0
                    for i in range(-5, 1):
                        t = zz + i
                        for j in range(-10, 1):
                            cumul += a[__an + i, __bn + j] + t
                    __b0[__an, __bn] = cumul / 50
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        nh = ((-5, 0), (-10, 0))
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic71(self):
        """neighborhood, type change"""

        def kernel(a):
            cumul = 0
            for i in range(-29, 1):
                k = 0.0
                if i > -15:
                    k = 1j
                cumul += a[i] + k
            return cumul / 30

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __an in range(29, a.shape[0]):
                cumul = 0
                for i in range(-29, 1):
                    k = 0.0
                    if i > -15:
                        k = 1j
                    cumul += a[__an + i] + k
                __b0[__an,] = cumul / 30
            return __b0
        a = np.arange(60.0)
        nh = ((-29, 0),)
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic72(self):
        """neighborhood, narrower range than specified"""

        def kernel(a):
            cumul = 0
            for i in range(-19, -3):
                cumul += a[i]
            return cumul / 30

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __an in range(29, a.shape[0]):
                cumul = 0
                for i in range(-19, -3):
                    cumul += a[__an + i]
                __b0[__an,] = cumul / 30
            return __b0
        a = np.arange(60.0)
        nh = ((-29, 0),)
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic73(self):
        """neighborhood, +ve range"""

        def kernel(a):
            cumul = 0
            for i in range(5, 11):
                cumul += a[i]
            return cumul / 30

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __an in range(0, a.shape[0] - 10):
                cumul = 0
                for i in range(5, 11):
                    cumul += a[__an + i]
                __b0[__an,] = cumul / 30
            return __b0
        a = np.arange(60.0)
        nh = ((5, 10),)
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic73b(self):
        """neighborhood, -ve range"""

        def kernel(a):
            cumul = 0
            for i in range(-10, -4):
                cumul += a[i]
            return cumul / 30

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __an in range(10, a.shape[0]):
                cumul = 0
                for i in range(-10, -4):
                    cumul += a[__an + i]
                __b0[__an,] = cumul / 30
            return __b0
        a = np.arange(60.0)
        nh = ((-10, -5),)
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic74(self):
        """neighborhood, -ve->+ve range span"""

        def kernel(a):
            cumul = 0
            for i in range(-5, 11):
                cumul += a[i]
            return cumul / 30

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __an in range(5, a.shape[0] - 10):
                cumul = 0
                for i in range(-5, 11):
                    cumul += a[__an + i]
                __b0[__an,] = cumul / 30
            return __b0
        a = np.arange(60.0)
        nh = ((-5, 10),)
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic75(self):
        """neighborhood, -ve->-ve range span"""

        def kernel(a):
            cumul = 0
            for i in range(-10, -1):
                cumul += a[i]
            return cumul / 30

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __an in range(10, a.shape[0]):
                cumul = 0
                for i in range(-10, -1):
                    cumul += a[__an + i]
                __b0[__an,] = cumul / 30
            return __b0
        a = np.arange(60.0)
        nh = ((-10, -2),)
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic76(self):
        """neighborhood, mixed range span"""

        def kernel(a):
            cumul = 0
            zz = 12.0
            for i in range(-3, 0):
                t = zz + i
                for j in range(-3, 4):
                    cumul += a[i, j] + t
            return cumul / (10 * 5)

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __bn in range(3, a.shape[1] - 3):
                for __an in range(3, a.shape[0]):
                    cumul = 0
                    zz = 12.0
                    for i in range(-3, 0):
                        t = zz + i
                        for j in range(-3, 4):
                            cumul += a[__an + i, __bn + j] + t
                    __b0[__an, __bn] = cumul / 50
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        nh = ((-3, -1), (-3, 3))
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    def test_basic77(self):
        """ neighborhood, two args """

        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[i, j]
            return cumul / 9.0

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, b, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __bn in range(3, a.shape[1]):
                for __an in range(3, a.shape[0]):
                    cumul = 0
                    for i in range(-3, 1):
                        for j in range(-3, 1):
                            cumul += a[__an + i, __bn + j] + b[__an + i, __bn + j]
                    __b0[__an, __bn] = cumul / 9.0
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        b = np.arange(10.0 * 20.0).reshape(10, 20)
        nh = ((-3, 0), (-3, 0))
        expected = __kernel(a, b, nh)
        self.check_against_expected(kernel, expected, a, b, options={'neighborhood': nh})

    def test_basic78(self):
        """ neighborhood, two args, -ve range, -ve range """

        def kernel(a, b):
            cumul = 0
            for i in range(-6, -2):
                for j in range(-7, -1):
                    cumul += a[i, j] + b[i, j]
            return cumul / 9.0

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, b, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __bn in range(7, a.shape[1]):
                for __an in range(6, a.shape[0]):
                    cumul = 0
                    for i in range(-6, -2):
                        for j in range(-7, -1):
                            cumul += a[__an + i, __bn + j] + b[__an + i, __bn + j]
                    __b0[__an, __bn] = cumul / 9.0
            return __b0
        a = np.arange(15.0 * 20.0).reshape(15, 20)
        b = np.arange(15.0 * 20.0).reshape(15, 20)
        nh = ((-6, -3), (-7, -2))
        expected = __kernel(a, b, nh)
        self.check_against_expected(kernel, expected, a, b, options={'neighborhood': nh})

    def test_basic78b(self):
        """ neighborhood, two args, -ve range, +ve range """

        def kernel(a, b):
            cumul = 0
            for i in range(-6, -2):
                for j in range(2, 10):
                    cumul += a[i, j] + b[i, j]
            return cumul / 9.0

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, b, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __bn in range(0, a.shape[1] - 9):
                for __an in range(6, a.shape[0]):
                    cumul = 0
                    for i in range(-6, -2):
                        for j in range(2, 10):
                            cumul += a[__an + i, __bn + j] + b[__an + i, __bn + j]
                    __b0[__an, __bn] = cumul / 9.0
            return __b0
        a = np.arange(15.0 * 20.0).reshape(15, 20)
        b = np.arange(15.0 * 20.0).reshape(15, 20)
        nh = ((-6, -3), (2, 9))
        expected = __kernel(a, b, nh)
        self.check_against_expected(kernel, expected, a, b, options={'neighborhood': nh})

    def test_basic79(self):
        """ neighborhood, two incompatible args """

        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[i, j]
            return cumul / 9.0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        b = np.arange(10.0 * 20.0).reshape(10, 10, 2)
        ex = self.exception_dict(stencil=TypingError, parfor=TypingError, njit=TypingError)
        self.check_exceptions(kernel, a, b, options={'neighborhood': ((-3, 0), (-3, 0))}, expected_exception=ex)

    def test_basic80(self):
        """ neighborhood, type change """

        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b
            return cumul / 9.0

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, b, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __bn in range(3, a.shape[1]):
                for __an in range(3, a.shape[0]):
                    cumul = 0
                    for i in range(-3, 1):
                        for j in range(-3, 1):
                            cumul += a[__an + i, __bn + j] + b
                    __b0[__an, __bn] = cumul / 9.0
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        b = 12j
        nh = ((-3, 0), (-3, 0))
        expected = __kernel(a, b, nh)
        self.check_against_expected(kernel, expected, a, b, options={'neighborhood': nh})

    def test_basic81(self):
        """ neighborhood, dimensionally incompatible arrays """

        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[i]
            return cumul / 9.0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        b = a[0].copy()
        ex = self.exception_dict(stencil=TypingError, parfor=AssertionError, njit=TypingError)
        self.check_exceptions(kernel, a, b, options={'neighborhood': ((-3, 0), (-3, 0))}, expected_exception=ex)

    def test_basic82(self):
        """ neighborhood, with standard_indexing"""

        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[1, 3]
            return cumul / 9.0

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, b, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __bn in range(3, a.shape[1]):
                for __an in range(3, a.shape[0]):
                    cumul = 0
                    for i in range(-3, 1):
                        for j in range(-3, 1):
                            cumul += a[__an + i, __bn + j] + b[1, 3]
                    __b0[__an, __bn] = cumul / 9.0
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        b = a.copy()
        nh = ((-3, 0), (-3, 0))
        expected = __kernel(a, b, nh)
        self.check_against_expected(kernel, expected, a, b, options={'neighborhood': nh, 'standard_indexing': 'b'})

    def test_basic83(self):
        """ neighborhood, with standard_indexing and cval"""

        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[1, 3]
            return cumul / 9.0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        b = a.copy()

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, b, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 1.5, dtype=type(__retdtype))
            for __bn in range(3, a.shape[1]):
                for __an in range(3, a.shape[0]):
                    cumul = 0
                    for i in range(-3, 1):
                        for j in range(-3, 1):
                            cumul += a[__an + i, __bn + j] + b[1, 3]
                    __b0[__an, __bn] = cumul / 9.0
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        b = a.copy()
        nh = ((-3, 0), (-3, 0))
        expected = __kernel(a, b, nh)
        self.check_against_expected(kernel, expected, a, b, options={'neighborhood': nh, 'standard_indexing': 'b', 'cval': 1.5})

    def test_basic84(self):
        """ kernel calls njit """

        def kernel(a):
            return a[0, 0] + addone_njit(a[0, 1])

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + addone_njit.py_func(a[__a + 0, __b + 1])
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic85(self):
        """ kernel calls njit(parallel=True)"""

        def kernel(a):
            return a[0, 0] + addone_pjit(a[0, 1])

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 1):
                for __a in range(0, a.shape[0]):
                    __b0[__a, __b] = a[__a + 0, __b + 0] + addone_pjit.py_func(a[__a + 0, __b + 1])
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic86(self):
        """ bad kwarg """

        def kernel(a):
            return a[0, 0]
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        self.check_exceptions(kernel, a, options={'bad': 10}, expected_exception=[ValueError, TypingError])

    def test_basic87(self):
        """ reserved arg name in use """

        def kernel(__sentinel__):
            return __sentinel__[0, 0]

        def __kernel(__sentinel__, neighborhood):
            self.check_stencil_arrays(__sentinel__, neighborhood=neighborhood)
            __retdtype = kernel(__sentinel__)
            __b0 = np.full(__sentinel__.shape, 0, dtype=type(__retdtype))
            for __b in range(0, __sentinel__.shape[1]):
                for __a in range(0, __sentinel__.shape[0]):
                    __b0[__a, __b] = __sentinel__[__a + 0, __b + 0]
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic88(self):
        """ use of reserved word """

        def kernel(a, out):
            return out * a[0, 1]
        a = np.arange(12.0).reshape(3, 4)
        ex = self.exception_dict(stencil=NumbaValueError, parfor=ValueError, njit=NumbaValueError)
        self.check_exceptions(kernel, a, 1.0, options={}, expected_exception=ex)

    def test_basic89(self):
        """ basic multiple return"""

        def kernel(a):
            if a[0, 1] > 10:
                return 10.0
            elif a[0, 3] < 8:
                return a[0, 0]
            else:
                return 7.0

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1] - 3):
                for __a in range(0, a.shape[0]):
                    if a[__a + 0, __b + 1] > 10:
                        __b0[__a, __b] = 10.0
                    elif a[__a + 0, __b + 3] < 8:
                        __b0[__a, __b] = a[__a + 0, __b + 0]
                    else:
                        __b0[__a, __b] = 7.0
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic90(self):
        """ neighborhood, with standard_indexing and cval, multiple returns"""

        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[1, 3]
            res = cumul / 9.0
            if res > 200.0:
                return res + 1.0
            else:
                return res

        def __kernel(a, b, neighborhood):
            self.check_stencil_arrays(a, b, neighborhood=neighborhood)
            __retdtype = kernel(a, b)
            __b0 = np.full(a.shape, 1.5, dtype=type(__retdtype))
            for __bn in range(3, a.shape[1]):
                for __an in range(3, a.shape[0]):
                    cumul = 0
                    for i in range(-3, 1):
                        for j in range(-3, 1):
                            cumul += a[__an + i, __bn + j] + b[1, 3]
                    res = cumul / 9.0
                    if res > 200.0:
                        __b0[__an, __bn] = res + 1.0
                    else:
                        __b0[__an, __bn] = res
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        b = a.copy()
        nh = ((-3, 0), (-3, 0))
        expected = __kernel(a, b, nh)
        self.check_against_expected(kernel, expected, a, b, options={'neighborhood': nh, 'standard_indexing': 'b', 'cval': 1.5})

    def test_basic91(self):
        """ Issue #3454, const(int) == const(int) evaluating incorrectly. """

        def kernel(a):
            b = 0
            if 2 == 0:
                b = 2
            return a[0, 0] + b

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(0, a.shape[1]):
                for __a in range(0, a.shape[0]):
                    b = 0
                    if 2 == 0:
                        b = 2
                    __b0[__a, __b] = a[__a + 0, __b + 0] + b
            return __b0
        a = np.arange(10.0 * 20.0).reshape(10, 20)
        expected = __kernel(a, None)
        self.check_against_expected(kernel, expected, a)

    def test_basic92(self):
        """ Issue #3497, bool return type evaluating incorrectly. """

        def kernel(a):
            return a[-1, -1] ^ a[-1, 0] ^ a[-1, 1] ^ a[0, -1] ^ a[0, 0] ^ a[0, 1] ^ a[1, -1] ^ a[1, 0] ^ a[1, 1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __b in range(1, a.shape[1] - 1):
                for __a in range(1, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + -1, __b + -1] ^ a[__a + -1, __b + 0] ^ a[__a + -1, __b + 1] ^ a[__a + 0, __b + -1] ^ a[__a + 0, __b + 0] ^ a[__a + 0, __b + 1] ^ a[__a + 1, __b + -1] ^ a[__a + 1, __b + 0] ^ a[__a + 1, __b + 1]
            return __b0
        A = np.array(np.arange(20) % 2).reshape(4, 5).astype(np.bool_)
        expected = __kernel(A, None)
        self.check_against_expected(kernel, expected, A)

    def test_basic93(self):
        """ Issue #3497, bool return type evaluating incorrectly. """

        def kernel(a):
            return a[-1, -1] ^ a[-1, 0] ^ a[-1, 1] ^ a[0, -1] ^ a[0, 0] ^ a[0, 1] ^ a[1, -1] ^ a[1, 0] ^ a[1, 1]

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 1, dtype=type(__retdtype))
            for __b in range(1, a.shape[1] - 1):
                for __a in range(1, a.shape[0] - 1):
                    __b0[__a, __b] = a[__a + -1, __b + -1] ^ a[__a + -1, __b + 0] ^ a[__a + -1, __b + 1] ^ a[__a + 0, __b + -1] ^ a[__a + 0, __b + 0] ^ a[__a + 0, __b + 1] ^ a[__a + 1, __b + -1] ^ a[__a + 1, __b + 0] ^ a[__a + 1, __b + 1]
            return __b0
        A = np.array(np.arange(20) % 2).reshape(4, 5).astype(np.bool_)
        expected = __kernel(A, None)
        self.check_against_expected(kernel, expected, A, options={'cval': True})

    def test_basic94(self):
        """ Issue #3528. Support for slices. """

        def kernel(a):
            return np.median(a[-1:2, -1:2])

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __bn in range(1, a.shape[1] - 1):
                for __an in range(1, a.shape[0] - 1):
                    __b0[__an, __bn] = np.median(a[__an + -1:__an + 2, __bn + -1:__bn + 2])
            return __b0
        a = np.arange(20, dtype=np.uint32).reshape(4, 5)
        nh = ((-1, 1), (-1, 1))
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    @unittest.skip('not yet supported')
    def test_basic95(self):
        """ Slice, calculate neighborhood. """

        def kernel(a):
            return np.median(a[-1:2, -3:4])

    def test_basic96(self):
        """ 1D slice. """

        def kernel(a):
            return np.median(a[-1:2])

        def __kernel(a, neighborhood):
            self.check_stencil_arrays(a, neighborhood=neighborhood)
            __retdtype = kernel(a)
            __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
            for __an in range(1, a.shape[0] - 1):
                __b0[__an,] = np.median(a[__an + -1:__an + 2])
            return __b0
        a = np.arange(20, dtype=np.uint32)
        nh = ((-1, 1),)
        expected = __kernel(a, nh)
        self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})

    @unittest.skip('not yet supported')
    def test_basic97(self):
        """ 2D slice and index. """

        def kernel(a):
            return np.median(a[-1:2, 3])

    def test_basic98(self):
        """ Test issue #7286 where the cval is a np attr/string-based numerical
        constant"""
        for cval in (np.nan, np.inf, -np.inf, float('inf'), -float('inf')):

            def kernel(a):
                return a[0, 0]

            def __kernel(a, neighborhood):
                self.check_stencil_arrays(a, neighborhood=neighborhood)
                __retdtype = kernel(a)
                __b0 = np.full(a.shape, cval, dtype=type(__retdtype))
                for __bn in range(1, a.shape[1] - 1):
                    for __an in range(1, a.shape[0] - 1):
                        __b0[__an, __bn] = a[__an + 0, __bn + 0]
                return __b0
            a = np.arange(6.0).reshape((2, 3))
            nh = ((-1, 1), (-1, 1))
            expected = __kernel(a, nh)
            self.check_against_expected(kernel, expected, a, options={'neighborhood': nh, 'cval': cval})