import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
class TestHighLevelExtending(TestCase):
    """
    Test the high-level combined API.
    """

    def test_where(self):
        """
        Test implementing a function with @overload.
        """
        pyfunc = call_where
        cfunc = jit(nopython=True)(pyfunc)

        def check(*args, **kwargs):
            expected = np_where(*args, **kwargs)
            got = cfunc(*args, **kwargs)
            self.assertPreciseEqual(expected, got)
        check(x=3, cond=True, y=8)
        check(True, 3, 8)
        check(np.bool_([True, False, True]), np.int32([1, 2, 3]), np.int32([4, 5, 5]))
        with self.assertRaises(errors.TypingError) as raises:
            cfunc(np.bool_([]), np.int32([]), np.int64([]))
        self.assertIn('x and y should have the same dtype', str(raises.exception))

    def test_len(self):
        """
        Test re-implementing len() for a custom type with @overload.
        """
        cfunc = jit(nopython=True)(len_usecase)
        self.assertPreciseEqual(cfunc(MyDummy()), 13)
        self.assertPreciseEqual(cfunc([4, 5]), 2)

    def test_print(self):
        """
        Test re-implementing print() for a custom type with @overload.
        """
        cfunc = jit(nopython=True)(print_usecase)
        with captured_stdout():
            cfunc(MyDummy())
            self.assertEqual(sys.stdout.getvalue(), 'hello!\n')

    def test_add_operator(self):
        """
        Test re-implementing operator.add() for a custom type with @overload.
        """
        pyfunc = call_add_operator
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(1, 2), 3)
        self.assertPreciseEqual(cfunc(MyDummy2(), MyDummy2()), 42)
        self.assertPreciseEqual(cfunc(MyDummy(), MyDummy()), 84)

    def test_add_binop(self):
        """
        Test re-implementing '+' for a custom type via @overload(operator.add).
        """
        pyfunc = call_add_binop
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(1, 2), 3)
        self.assertPreciseEqual(cfunc(MyDummy2(), MyDummy2()), 42)
        self.assertPreciseEqual(cfunc(MyDummy(), MyDummy()), 84)

    def test_iadd_operator(self):
        """
        Test re-implementing operator.add() for a custom type with @overload.
        """
        pyfunc = call_iadd_operator
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(1, 2), 3)
        self.assertPreciseEqual(cfunc(MyDummy2(), MyDummy2()), 42)
        self.assertPreciseEqual(cfunc(MyDummy(), MyDummy()), 84)

    def test_iadd_binop(self):
        """
        Test re-implementing '+' for a custom type via @overload(operator.add).
        """
        pyfunc = call_iadd_binop
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(1, 2), 3)
        self.assertPreciseEqual(cfunc(MyDummy2(), MyDummy2()), 42)
        self.assertPreciseEqual(cfunc(MyDummy(), MyDummy()), 84)

    def test_delitem(self):
        pyfunc = call_delitem
        cfunc = jit(nopython=True)(pyfunc)
        obj = MyDummy()
        e = None
        with captured_stdout() as out:
            try:
                cfunc(obj, 321)
            except Exception as exc:
                e = exc
        if e is not None:
            raise e
        self.assertEqual(out.getvalue(), 'del hello! 321\n')

    def test_getitem(self):
        pyfunc = call_getitem
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(MyDummy(), 321), 321 + 123)

    def test_setitem(self):
        pyfunc = call_setitem
        cfunc = jit(nopython=True)(pyfunc)
        obj = MyDummy()
        e = None
        with captured_stdout() as out:
            try:
                cfunc(obj, 321, 123)
            except Exception as exc:
                e = exc
        if e is not None:
            raise e
        self.assertEqual(out.getvalue(), '321 123\n')

    def test_no_cpython_wrapper(self):
        """
        Test overloading whose return value cannot be represented in CPython.
        """
        ok_cfunc = jit(nopython=True)(non_boxable_ok_usecase)
        n = 10
        got = ok_cfunc(n)
        expect = non_boxable_ok_usecase(n)
        np.testing.assert_equal(expect, got)
        bad_cfunc = jit(nopython=True)(non_boxable_bad_usecase)
        with self.assertRaises(TypeError) as raises:
            bad_cfunc()
        errmsg = str(raises.exception)
        expectmsg = 'cannot convert native Module'
        self.assertIn(expectmsg, errmsg)

    def test_typing_vs_impl_signature_mismatch_handling(self):
        """
        Tests that an overload which has a differing typing and implementing
        signature raises an exception.
        """

        def gen_ol(impl=None):

            def myoverload(a, b, c, kw=None):
                pass

            @overload(myoverload)
            def _myoverload_impl(a, b, c, kw=None):
                return impl

            @jit(nopython=True)
            def foo(a, b, c, d):
                myoverload(a, b, c, kw=d)
            return foo
        sentinel = 'Typing and implementation arguments differ in'

        def impl1(a, b, c, kw=12):
            if a > 10:
                return 1
            else:
                return -1
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl1)(1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn('keyword argument default values', msg)
        self.assertIn('<Parameter "kw=12">', msg)
        self.assertIn('<Parameter "kw=None">', msg)

        def impl2(a, b, c, kwarg=None):
            if a > 10:
                return 1
            else:
                return -1
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl2)(1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn('keyword argument names', msg)
        self.assertIn('<Parameter "kwarg=None">', msg)
        self.assertIn('<Parameter "kw=None">', msg)

        def impl3(z, b, c, kw=None):
            if a > 10:
                return 1
            else:
                return -1
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl3)(1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn('argument names', msg)
        self.assertFalse('keyword' in msg)
        self.assertIn('<Parameter "a">', msg)
        self.assertIn('<Parameter "z">', msg)
        from .overload_usecases import impl4, impl5
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl4)(1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn('argument names', msg)
        self.assertFalse('keyword' in msg)
        self.assertIn("First difference: 'z'", msg)
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl5)(1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn('argument names', msg)
        self.assertFalse('keyword' in msg)
        self.assertIn('<Parameter "a">', msg)
        self.assertIn('<Parameter "z">', msg)

        def impl6(a, b, c, d, e, kw=None):
            if a > 10:
                return 1
            else:
                return -1
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl6)(1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn('argument names', msg)
        self.assertFalse('keyword' in msg)
        self.assertIn('<Parameter "d">', msg)
        self.assertIn('<Parameter "e">', msg)

        def impl7(a, b, kw=None):
            if a > 10:
                return 1
            else:
                return -1
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl7)(1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn('argument names', msg)
        self.assertFalse('keyword' in msg)
        self.assertIn('<Parameter "c">', msg)

        def impl8(a, b, c, kw=None, extra_kwarg=None):
            if a > 10:
                return 1
            else:
                return -1
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl8)(1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn('keyword argument names', msg)
        self.assertIn('<Parameter "extra_kwarg=None">', msg)

        def impl9(a, b, c):
            if a > 10:
                return 1
            else:
                return -1
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl9)(1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn('keyword argument names', msg)
        self.assertIn('<Parameter "kw=None">', msg)

    def test_typing_vs_impl_signature_mismatch_handling_var_positional(self):
        """
        Tests that an overload which has a differing typing and implementing
        signature raises an exception and uses VAR_POSITIONAL (*args) in typing
        """

        def myoverload(a, kw=None):
            pass
        from .overload_usecases import var_positional_impl
        overload(myoverload)(var_positional_impl)

        @jit(nopython=True)
        def foo(a, b):
            return myoverload(a, b, 9, kw=11)
        with self.assertRaises(errors.TypingError) as e:
            foo(1, 5)
        msg = str(e.exception)
        self.assertIn('VAR_POSITIONAL (e.g. *args) argument kind', msg)
        self.assertIn("offending argument name is '*star_args_token'", msg)

    def test_typing_vs_impl_signature_mismatch_handling_var_keyword(self):
        """
        Tests that an overload which uses **kwargs (VAR_KEYWORD)
        """

        def gen_ol(impl, strict=True):

            def myoverload(a, kw=None):
                pass
            overload(myoverload, strict=strict)(impl)

            @jit(nopython=True)
            def foo(a, b):
                return myoverload(a, kw=11)
            return foo

        def ol1(a, **kws):

            def impl(a, kw=10):
                return a
            return impl
        gen_ol(ol1, False)(1, 2)
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(ol1)(1, 2)
        msg = str(e.exception)
        self.assertIn('use of VAR_KEYWORD (e.g. **kwargs) is unsupported', msg)
        self.assertIn("offending argument name is '**kws'", msg)

        def ol2(a, kw=0):

            def impl(a, **kws):
                return a
            return impl
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(ol2)(1, 2)
        msg = str(e.exception)
        self.assertIn('use of VAR_KEYWORD (e.g. **kwargs) is unsupported', msg)
        self.assertIn("offending argument name is '**kws'", msg)

    def test_overload_method_kwargs(self):

        @overload_method(types.Array, 'foo')
        def fooimpl(arr, a_kwarg=10):

            def impl(arr, a_kwarg=10):
                return a_kwarg
            return impl

        @njit
        def bar(A):
            return (A.foo(), A.foo(20), A.foo(a_kwarg=30))
        Z = np.arange(5)
        self.assertEqual(bar(Z), (10, 20, 30))

    def test_overload_method_literal_unpack(self):

        @overload_method(types.Array, 'litfoo')
        def litfoo(arr, val):
            if isinstance(val, types.Integer):
                if not isinstance(val, types.Literal):

                    def impl(arr, val):
                        return val
                    return impl

        @njit
        def bar(A):
            return A.litfoo(51966)
        A = np.zeros(1)
        bar(A)
        self.assertEqual(bar(A), 51966)

    def test_overload_ufunc(self):

        @njit
        def test():
            return np.exp(mydummy)
        self.assertEqual(test(), 3735928559)

    def test_overload_method_stararg(self):

        @overload_method(MyDummyType, 'method_stararg')
        def _ov_method_stararg(obj, val, val2, *args):

            def get(obj, val, val2, *args):
                return (val, val2, args)
            return get

        @njit
        def foo(obj, *args):
            return obj.method_stararg(*args)
        obj = MyDummy()
        self.assertEqual(foo(obj, 1, 2), (1, 2, ()))
        self.assertEqual(foo(obj, 1, 2, 3), (1, 2, (3,)))
        self.assertEqual(foo(obj, 1, 2, 3, 4), (1, 2, (3, 4)))

        @njit
        def bar(obj):
            return (obj.method_stararg(1, 2), obj.method_stararg(1, 2, 3), obj.method_stararg(1, 2, 3, 4))
        self.assertEqual(bar(obj), ((1, 2, ()), (1, 2, (3,)), (1, 2, (3, 4))))
        self.assertEqual(foo(obj, 1, 2, (3,)), (1, 2, ((3,),)))
        self.assertEqual(foo(obj, 1, 2, (3, 4)), (1, 2, ((3, 4),)))
        self.assertEqual(foo(obj, 1, 2, (3, (4, 5))), (1, 2, ((3, (4, 5)),)))

    def test_overload_classmethod(self):

        class MyArray(types.Array):
            pass

        @overload_classmethod(MyArray, 'array_alloc')
        def ol_array_alloc(cls, nitems):

            def impl(cls, nitems):
                arr = np.arange(nitems)
                return arr
            return impl

        @njit
        def foo(nitems):
            return MyArray.array_alloc(nitems)
        nitems = 13
        self.assertPreciseEqual(foo(nitems), np.arange(nitems))

        @njit
        def no_classmethod_in_base(nitems):
            return types.Array.array_alloc(nitems)
        with self.assertRaises(errors.TypingError) as raises:
            no_classmethod_in_base(nitems)
        self.assertIn("Unknown attribute 'array_alloc' of", str(raises.exception))

    def test_overload_callable_typeref(self):

        @overload(CallableTypeRef)
        def callable_type_call_ovld1(x):
            if isinstance(x, types.Integer):

                def impl(x):
                    return 42.5 + x
                return impl

        @overload(CallableTypeRef)
        def callable_type_call_ovld2(x):
            if isinstance(x, types.UnicodeType):

                def impl(x):
                    return '42.5' + x
                return impl

        @njit
        def foo(a, b):
            return (MyClass(a), MyClass(b))
        args = (4, '4')
        expected = (42.5 + args[0], '42.5' + args[1])
        self.assertPreciseEqual(foo(*args), expected)