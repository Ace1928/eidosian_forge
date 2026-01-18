import sys
import copy
import logging
import numpy as np
from numba import njit, jit, types
from numba.core import errors, ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.untyped_passes import ReconstructSSA, PreserveIR
from numba.core.typed_passes import NativeLowering
from numba.extending import overload
from numba.tests.support import MemoryLeakMixin, TestCase, override_config
class TestSSA(SSABaseTest):
    """
    Contains tests to help isolate problems in SSA
    """

    def test_argument_name_reused(self):

        @njit
        def foo(x):
            x += 1
            return x
        self.check_func(foo, 123)

    def test_if_else_redefine(self):

        @njit
        def foo(x, y):
            z = x * y
            if x < y:
                z = x
            else:
                z = y
            return z
        self.check_func(foo, 3, 2)
        self.check_func(foo, 2, 3)

    def test_sum_loop(self):

        @njit
        def foo(n):
            c = 0
            for i in range(n):
                c += i
            return c
        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def test_sum_loop_2vars(self):

        @njit
        def foo(n):
            c = 0
            d = n
            for i in range(n):
                c += i
                d += n
            return (c, d)
        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def test_sum_2d_loop(self):

        @njit
        def foo(n):
            c = 0
            for i in range(n):
                for j in range(n):
                    c += j
                c += i
            return c
        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def check_undefined_var(self, should_warn):

        @njit
        def foo(n):
            if n:
                if n > 0:
                    c = 0
                return c
            else:
                c += 1
                return c
        if should_warn:
            with self.assertWarns(errors.NumbaWarning) as warns:
                self.check_func(foo, 1)
            self.assertIn('Detected uninitialized variable c', str(warns.warning))
        else:
            self.check_func(foo, 1)
        with self.assertRaises(UnboundLocalError):
            foo.py_func(0)

    def test_undefined_var(self):
        with override_config('ALWAYS_WARN_UNINIT_VAR', 0):
            self.check_undefined_var(should_warn=False)
        with override_config('ALWAYS_WARN_UNINIT_VAR', 1):
            self.check_undefined_var(should_warn=True)

    def test_phi_propagation(self):

        @njit
        def foo(actions):
            n = 1
            i = 0
            ct = 0
            while n > 0 and i < len(actions):
                n -= 1
                while actions[i]:
                    if actions[i]:
                        if actions[i]:
                            n += 10
                        actions[i] -= 1
                    else:
                        if actions[i]:
                            n += 20
                        actions[i] += 1
                    ct += n
                ct += n
            return (ct, n)
        self.check_func(foo, np.array([1, 2]))

    def test_unhandled_undefined(self):

        def function1(arg1, arg2, arg3, arg4, arg5):
            if arg1:
                var1 = arg2
                var2 = arg3
                var3 = var2
                var4 = arg1
                return
            else:
                if arg2:
                    if arg4:
                        var5 = arg4
                        return
                    else:
                        var6 = var4
                        return
                    return var6
                else:
                    if arg5:
                        if var1:
                            if arg5:
                                var1 = var6
                                return
                            else:
                                var7 = arg2
                                return arg2
                            return
                        else:
                            if var2:
                                arg5 = arg2
                                return arg1
                            else:
                                var6 = var3
                                return var4
                            return
                        return
                    else:
                        var8 = var1
                        return
                    return var8
                var9 = var3
                var10 = arg5
                return var1
        expect = function1(2, 3, 6, 0, 7)
        got = njit(function1)(2, 3, 6, 0, 7)
        self.assertEqual(expect, got)