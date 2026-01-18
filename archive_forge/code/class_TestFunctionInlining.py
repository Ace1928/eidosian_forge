import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
class TestFunctionInlining(MemoryLeakMixin, InliningBase):

    def test_basic_inline_never(self):

        @njit(inline='never')
        def foo():
            return

        def impl():
            return foo()
        self.check(impl, inline_expect={'foo': False})

    def test_basic_inline_always(self):

        @njit(inline='always')
        def foo():
            return

        def impl():
            return foo()
        self.check(impl, inline_expect={'foo': True})

    def test_basic_inline_combos(self):

        def impl():
            x = foo()
            y = bar()
            z = baz()
            return (x, y, z)
        opts = ('always', 'never')
        for inline_foo, inline_bar, inline_baz in product(opts, opts, opts):

            @njit(inline=inline_foo)
            def foo():
                return

            @njit(inline=inline_bar)
            def bar():
                return

            @njit(inline=inline_baz)
            def baz():
                return
            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo], 'bar': self.inline_opt_as_bool[inline_bar], 'baz': self.inline_opt_as_bool[inline_baz]}
            self.check(impl, inline_expect=inline_expect)

    @unittest.skip('Need to work out how to prevent this')
    def test_recursive_inline(self):

        @njit(inline='always')
        def foo(x):
            if x == 0:
                return 12
            else:
                foo(x - 1)
        a = 3

        def impl():
            b = 0
            if a > 1:
                b += 1
            foo(5)
            if b < a:
                b -= 1
        self.check(impl, inline_expect={'foo': True})

    def test_freevar_bindings(self):

        def factory(inline, x, y):
            z = x + 12

            @njit(inline=inline)
            def func():
                return (x, y + 3, z)
            return func

        def impl():
            x = foo()
            y = bar()
            z = baz()
            return (x, y, z)
        opts = ('always', 'never')
        for inline_foo, inline_bar, inline_baz in product(opts, opts, opts):
            foo = factory(inline_foo, 10, 20)
            bar = factory(inline_bar, 30, 40)
            baz = factory(inline_baz, 50, 60)
            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo], 'bar': self.inline_opt_as_bool[inline_bar], 'baz': self.inline_opt_as_bool[inline_baz]}
            self.check(impl, inline_expect=inline_expect)

    def test_global_binding(self):

        def impl():
            x = 19
            return _global_func(x)
        self.check(impl, inline_expect={'_global_func': True})

    def test_inline_from_another_module(self):
        from .inlining_usecases import bar

        def impl():
            z = _GLOBAL1 + 2
            return (bar(), z)
        self.check(impl, inline_expect={'bar': True})

    def test_inline_from_another_module_w_getattr(self):
        import numba.tests.inlining_usecases as iuc

        def impl():
            z = _GLOBAL1 + 2
            return (iuc.bar(), z)
        self.check(impl, inline_expect={'bar': True})

    def test_inline_from_another_module_w_2_getattr(self):
        import numba.tests.inlining_usecases
        import numba.tests as nt

        def impl():
            z = _GLOBAL1 + 2
            return (nt.inlining_usecases.bar(), z)
        self.check(impl, inline_expect={'bar': True})

    def test_inline_from_another_module_as_freevar(self):

        def factory():
            from .inlining_usecases import bar

            @njit(inline='always')
            def tmp():
                return bar()
            return tmp
        baz = factory()

        def impl():
            z = _GLOBAL1 + 2
            return (baz(), z)
        self.check(impl, inline_expect={'bar': True})

    def test_inline_w_freevar_from_another_module(self):
        from .inlining_usecases import baz_factory

        def gen(a, b):
            bar = baz_factory(a)

            def impl():
                z = _GLOBAL1 + a * b
                return (bar(), z, a)
            return impl
        impl = gen(10, 20)
        self.check(impl, inline_expect={'bar': True})

    def test_inlining_models(self):

        def s17_caller_model(expr, caller_info, callee_info):
            self.assertIsInstance(expr, ir.Expr)
            self.assertEqual(expr.op, 'call')
            return self.sentinel_17_cost_model(caller_info)

        def s17_callee_model(expr, caller_info, callee_info):
            self.assertIsInstance(expr, ir.Expr)
            self.assertEqual(expr.op, 'call')
            return self.sentinel_17_cost_model(callee_info)
        for caller, callee in ((11, 17), (17, 11)):

            @njit(inline=s17_caller_model)
            def foo():
                return callee

            def impl(z):
                x = z + caller
                y = foo()
                return (y + 3, x)
            self.check(impl, 10, inline_expect={'foo': caller == 17})
        for caller, callee in ((11, 17), (17, 11)):

            @njit(inline=s17_callee_model)
            def bar():
                return callee

            def impl(z):
                x = z + caller
                y = bar()
                return (y + 3, x)
            self.check(impl, 10, inline_expect={'bar': callee == 17})

    def test_inline_inside_loop(self):

        @njit(inline='always')
        def foo():
            return 12

        def impl():
            acc = 0.0
            for i in range(5):
                acc += foo()
            return acc
        self.check(impl, inline_expect={'foo': True}, block_count=4)

    def test_inline_inside_closure_inside_loop(self):

        @njit(inline='always')
        def foo():
            return 12

        def impl():
            acc = 0.0
            for i in range(5):

                def bar():
                    return foo() + 7
                acc += bar()
            return acc
        self.check(impl, inline_expect={'foo': True}, block_count=4)

    def test_inline_closure_inside_inlinable_inside_closure(self):

        @njit(inline='always')
        def foo(a):

            def baz():
                return 12 + a
            return baz() + 8

        def impl():
            z = 9

            def bar(x):
                return foo(z) + 7 + x
            return bar(z + 2)
        self.check(impl, inline_expect={'foo': True}, block_count=1)

    def test_inline_involved(self):
        fortran = njit(inline='always')(_gen_involved())

        @njit(inline='always')
        def boz(j):
            acc = 0

            def biz(t):
                return t + acc
            for x in range(j):
                acc += biz(8 + acc) + fortran(2.0, acc, 1, 12j, biz(acc))
            return acc

        @njit(inline='always')
        def foo(a):
            acc = 0
            for p in range(12):
                tmp = fortran(1, 1, 1, 1, 1)

                def baz(x):
                    return 12 + a + x + tmp
                acc += baz(p) + 8 + boz(p) + tmp
            return acc + baz(2)

        def impl():
            z = 9

            def bar(x):
                return foo(z) + 7 + x
            return bar(z + 2)
        if utils.PYVERSION in ((3, 12),):
            bc = 39
        elif utils.PYVERSION in ((3, 10), (3, 11)):
            bc = 35
        elif utils.PYVERSION in ((3, 9),):
            bc = 33
        else:
            raise NotImplementedError(utils.PYVERSION)
        self.check(impl, inline_expect={'foo': True, 'boz': True, 'fortran': True}, block_count=bc)

    def test_inline_renaming_scheme(self):

        @njit(inline='always')
        def bar(z):
            x = 5
            y = 10
            return x + y + z

        @njit(pipeline_class=IRPreservingTestPipeline)
        def foo(a, b):
            return (bar(a), bar(b))
        self.assertEqual(foo(10, 20), (25, 35))
        func_ir = foo.overloads[foo.signatures[0]].metadata['preserved_ir']
        store = []
        for blk in func_ir.blocks.values():
            for stmt in blk.body:
                if isinstance(stmt, ir.Assign):
                    if isinstance(stmt.value, ir.Const):
                        if stmt.value.value == 5:
                            store.append(stmt)
        self.assertEqual(len(store), 2)
        for i in store:
            name = i.target.name
            basename = self.id().lstrip(self.__module__)
            regex = f'{basename}__locals__bar_v[0-9]+.x'
            self.assertRegex(name, regex)