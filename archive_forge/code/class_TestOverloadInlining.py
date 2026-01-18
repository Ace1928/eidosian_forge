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
class TestOverloadInlining(MemoryLeakMixin, InliningBase):

    def test_basic_inline_never(self):

        def foo():
            pass

        @overload(foo, inline='never')
        def foo_overload():

            def foo_impl():
                pass
            return foo_impl

        def impl():
            return foo()
        self.check(impl, inline_expect={'foo': False})

    def test_basic_inline_always(self):

        def foo():
            pass

        @overload(foo, inline='always')
        def foo_overload():

            def impl():
                pass
            return impl

        def impl():
            return foo()
        self.check(impl, inline_expect={'foo': True})

    def test_inline_always_kw_no_default(self):

        def foo(a, b):
            return a + b

        @overload(foo, inline='always')
        def overload_foo(a, b):
            return lambda a, b: a + b

        def impl():
            return foo(3, b=4)
        self.check(impl, inline_expect={'foo': True})

    def test_inline_operators_unary(self):

        def impl_inline(x):
            return -x

        def impl_noinline(x):
            return +x
        dummy_unary_impl = lambda x: True
        Dummy, DummyType = self.make_dummy_type()
        setattr(Dummy, '__neg__', dummy_unary_impl)
        setattr(Dummy, '__pos__', dummy_unary_impl)

        @overload(operator.neg, inline='always')
        def overload_dummy_neg(x):
            if isinstance(x, DummyType):
                return dummy_unary_impl

        @overload(operator.pos, inline='never')
        def overload_dummy_pos(x):
            if isinstance(x, DummyType):
                return dummy_unary_impl
        self.check(impl_inline, Dummy(), inline_expect={'neg': True})
        self.check(impl_noinline, Dummy(), inline_expect={'pos': False})

    def test_inline_operators_binop(self):

        def impl_inline(x):
            return x == 1

        def impl_noinline(x):
            return x != 1
        Dummy, DummyType = self.make_dummy_type()
        dummy_binop_impl = lambda a, b: True
        setattr(Dummy, '__eq__', dummy_binop_impl)
        setattr(Dummy, '__ne__', dummy_binop_impl)

        @overload(operator.eq, inline='always')
        def overload_dummy_eq(a, b):
            if isinstance(a, DummyType):
                return dummy_binop_impl

        @overload(operator.ne, inline='never')
        def overload_dummy_ne(a, b):
            if isinstance(a, DummyType):
                return dummy_binop_impl
        self.check(impl_inline, Dummy(), inline_expect={'eq': True})
        self.check(impl_noinline, Dummy(), inline_expect={'ne': False})

    def test_inline_operators_inplace_binop(self):

        def impl_inline(x):
            x += 1

        def impl_noinline(x):
            x -= 1
        Dummy, DummyType = self.make_dummy_type()
        dummy_inplace_binop_impl = lambda a, b: True
        setattr(Dummy, '__iadd__', dummy_inplace_binop_impl)
        setattr(Dummy, '__isub__', dummy_inplace_binop_impl)

        @overload(operator.iadd, inline='always')
        def overload_dummy_iadd(a, b):
            if isinstance(a, DummyType):
                return dummy_inplace_binop_impl

        @overload(operator.isub, inline='never')
        def overload_dummy_isub(a, b):
            if isinstance(a, DummyType):
                return dummy_inplace_binop_impl

        @overload(operator.add, inline='always')
        def overload_dummy_add(a, b):
            if isinstance(a, DummyType):
                return dummy_inplace_binop_impl

        @overload(operator.sub, inline='never')
        def overload_dummy_sub(a, b):
            if isinstance(a, DummyType):
                return dummy_inplace_binop_impl
        self.check(impl_inline, Dummy(), inline_expect={'iadd': True})
        self.check(impl_noinline, Dummy(), inline_expect={'isub': False})

    def test_inline_always_operators_getitem(self):

        def impl(x, idx):
            return x[idx]

        def impl_static_getitem(x):
            return x[1]
        Dummy, DummyType = self.make_dummy_type()
        dummy_getitem_impl = lambda obj, idx: None
        setattr(Dummy, '__getitem__', dummy_getitem_impl)

        @overload(operator.getitem, inline='always')
        def overload_dummy_getitem(obj, idx):
            if isinstance(obj, DummyType):
                return dummy_getitem_impl
        self.check(impl, Dummy(), 1, inline_expect={'getitem': True})
        self.check(impl_static_getitem, Dummy(), inline_expect={'getitem': True})

    def test_inline_never_operators_getitem(self):

        def impl(x, idx):
            return x[idx]

        def impl_static_getitem(x):
            return x[1]
        Dummy, DummyType = self.make_dummy_type()
        dummy_getitem_impl = lambda obj, idx: None
        setattr(Dummy, '__getitem__', dummy_getitem_impl)

        @overload(operator.getitem, inline='never')
        def overload_dummy_getitem(obj, idx):
            if isinstance(obj, DummyType):
                return dummy_getitem_impl
        self.check(impl, Dummy(), 1, inline_expect={'getitem': False})
        self.check(impl_static_getitem, Dummy(), inline_expect={'getitem': False})

    def test_inline_stararg_error(self):

        def foo(a, *b):
            return a + b[0]

        @overload(foo, inline='always')
        def overload_foo(a, *b):
            return lambda a, *b: a + b[0]

        def impl():
            return foo(3, 3, 5)
        with self.assertRaises(NotImplementedError) as e:
            self.check(impl, inline_expect={'foo': True})
        self.assertIn('Stararg not supported in inliner for arg 1 *b', str(e.exception))

    def test_basic_inline_combos(self):

        def impl():
            x = foo()
            y = bar()
            z = baz()
            return (x, y, z)
        opts = ('always', 'never')
        for inline_foo, inline_bar, inline_baz in product(opts, opts, opts):

            def foo():
                pass

            def bar():
                pass

            def baz():
                pass

            @overload(foo, inline=inline_foo)
            def foo_overload():

                def impl():
                    return
                return impl

            @overload(bar, inline=inline_bar)
            def bar_overload():

                def impl():
                    return
                return impl

            @overload(baz, inline=inline_baz)
            def baz_overload():

                def impl():
                    return
                return impl
            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo], 'bar': self.inline_opt_as_bool[inline_bar], 'baz': self.inline_opt_as_bool[inline_baz]}
            self.check(impl, inline_expect=inline_expect)

    def test_freevar_bindings(self):

        def impl():
            x = foo()
            y = bar()
            z = baz()
            return (x, y, z)
        opts = ('always', 'never')
        for inline_foo, inline_bar, inline_baz in product(opts, opts, opts):

            def foo():
                x = 10
                y = 20
                z = x + 12
                return (x, y + 3, z)

            def bar():
                x = 30
                y = 40
                z = x + 12
                return (x, y + 3, z)

            def baz():
                x = 60
                y = 80
                z = x + 12
                return (x, y + 3, z)

            def factory(target, x, y, inline=None):
                z = x + 12

                @overload(target, inline=inline)
                def func():

                    def impl():
                        return (x, y + 3, z)
                    return impl
            factory(foo, 10, 20, inline=inline_foo)
            factory(bar, 30, 40, inline=inline_bar)
            factory(baz, 60, 80, inline=inline_baz)
            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo], 'bar': self.inline_opt_as_bool[inline_bar], 'baz': self.inline_opt_as_bool[inline_baz]}
            self.check(impl, inline_expect=inline_expect)

    def test_global_overload_binding(self):

        def impl():
            z = 19
            return _global_defn(z)
        self.check(impl, inline_expect={'_global_defn': True})

    def test_inline_from_another_module(self):
        from .inlining_usecases import baz

        def impl():
            z = _GLOBAL1 + 2
            return (baz(), z)
        self.check(impl, inline_expect={'baz': True})

    def test_inline_from_another_module_w_getattr(self):
        import numba.tests.inlining_usecases as iuc

        def impl():
            z = _GLOBAL1 + 2
            return (iuc.baz(), z)
        self.check(impl, inline_expect={'baz': True})

    def test_inline_from_another_module_w_2_getattr(self):
        import numba.tests.inlining_usecases
        import numba.tests as nt

        def impl():
            z = _GLOBAL1 + 2
            return (nt.inlining_usecases.baz(), z)
        self.check(impl, inline_expect={'baz': True})

    def test_inline_from_another_module_as_freevar(self):

        def factory():
            from .inlining_usecases import baz

            @njit(inline='always')
            def tmp():
                return baz()
            return tmp
        bop = factory()

        def impl():
            z = _GLOBAL1 + 2
            return (bop(), z)
        self.check(impl, inline_expect={'baz': True})

    def test_inline_w_freevar_from_another_module(self):
        from .inlining_usecases import bop_factory

        def gen(a, b):
            bar = bop_factory(a)

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
            return self.sentinel_17_cost_model(caller_info.func_ir)

        def s17_callee_model(expr, caller_info, callee_info):
            self.assertIsInstance(expr, ir.Expr)
            self.assertEqual(expr.op, 'call')
            return self.sentinel_17_cost_model(callee_info.func_ir)
        for caller, callee in ((10, 11), (17, 11)):

            def foo():
                return callee

            @overload(foo, inline=s17_caller_model)
            def foo_ol():

                def impl():
                    return callee
                return impl

            def impl(z):
                x = z + caller
                y = foo()
                return (y + 3, x)
            self.check(impl, 10, inline_expect={'foo': caller == 17})
        for caller, callee in ((11, 17), (11, 10)):

            def bar():
                return callee

            @overload(bar, inline=s17_callee_model)
            def bar_ol():

                def impl():
                    return callee
                return impl

            def impl(z):
                x = z + caller
                y = bar()
                return (y + 3, x)
            self.check(impl, 10, inline_expect={'bar': callee == 17})

    def test_multiple_overloads_with_different_inline_characteristics(self):

        def bar(x):
            if isinstance(typeof(x), types.Float):
                return x + 1234
            else:
                return x + 1

        @overload(bar, inline='always')
        def bar_int_ol(x):
            if isinstance(x, types.Integer):

                def impl(x):
                    return x + 1
                return impl

        @overload(bar, inline='never')
        def bar_float_ol(x):
            if isinstance(x, types.Float):

                def impl(x):
                    return x + 1234
                return impl

        def always_inline_cost_model(*args):
            return True

        @overload(bar, inline=always_inline_cost_model)
        def bar_complex_ol(x):
            if isinstance(x, types.Complex):

                def impl(x):
                    return x + 1
                return impl

        def impl():
            a = bar(1)
            b = bar(2.3)
            c = bar(3j)
            return a + b + c
        fir = self.check(impl, inline_expect={'bar': False}, block_count=1)
        block = next(iter(fir.blocks.items()))[1]
        calls = [x for x in block.find_exprs(op='call')]
        self.assertTrue(len(calls) == 1)
        consts = [x.value for x in block.find_insts(ir.Assign) if isinstance(getattr(x, 'value', None), ir.Const)]
        for val in consts:
            self.assertNotEqual(val.value, 1234)

    def test_overload_inline_always_with_literally_in_inlinee(self):

        def foo_ovld(dtype):
            if not isinstance(dtype, types.StringLiteral):

                def foo_noop(dtype):
                    return literally(dtype)
                return foo_noop
            if dtype.literal_value == 'str':

                def foo_as_str_impl(dtype):
                    return 10
                return foo_as_str_impl
            if dtype.literal_value in ('int64', 'float64'):

                def foo_as_num_impl(dtype):
                    return 20
                return foo_as_num_impl

        def foo(dtype):
            return 10
        overload(foo, inline='always')(foo_ovld)

        def test_impl(dtype):
            return foo(dtype)
        dtype = 'str'
        self.check(test_impl, dtype, inline_expect={'foo': True})

        def foo(dtype):
            return 20
        overload(foo, inline='always')(foo_ovld)
        dtype = 'int64'
        self.check(test_impl, dtype, inline_expect={'foo': True})

    def test_inline_always_ssa(self):
        dummy_true = True

        def foo(A):
            return True

        @overload(foo, inline='always')
        def foo_overload(A):

            def impl(A):
                s = dummy_true
                for i in range(len(A)):
                    dummy = dummy_true
                    if A[i]:
                        dummy = A[i]
                    s *= dummy
                return s
            return impl

        def impl():
            return foo(np.array([True, False, True]))
        self.check(impl, block_count='SKIP', inline_expect={'foo': True})

    def test_inline_always_ssa_scope_validity(self):

        def bar():
            b = 5
            while b > 1:
                b //= 2
            return 10

        @overload(bar, inline='always')
        def bar_impl():
            return bar

        @njit
        def foo():
            bar()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', errors.NumbaIRAssumptionWarning)
            ignore_internal_warnings()
            self.assertEqual(foo(), foo.py_func())
        self.assertEqual(len(w), 0)