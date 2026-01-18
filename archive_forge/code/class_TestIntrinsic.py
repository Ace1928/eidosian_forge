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
class TestIntrinsic(TestCase):

    def test_void_return(self):
        """
        Verify that returning a None from codegen function is handled
        automatically for void functions, otherwise raise exception.
        """

        @intrinsic
        def void_func(typingctx, a):
            sig = types.void(types.int32)

            def codegen(context, builder, signature, args):
                pass
            return (sig, codegen)

        @intrinsic
        def non_void_func(typingctx, a):
            sig = types.int32(types.int32)

            def codegen(context, builder, signature, args):
                pass
            return (sig, codegen)

        @jit(nopython=True)
        def call_void_func():
            void_func(1)
            return 0

        @jit(nopython=True)
        def call_non_void_func():
            non_void_func(1)
            return 0
        self.assertEqual(call_void_func(), 0)
        with self.assertRaises(LoweringError) as e:
            call_non_void_func()
        self.assertIn('non-void function returns None', e.exception.msg)

    def test_ll_pointer_cast(self):
        """
        Usecase test: custom reinterpret cast to turn int values to pointers
        """
        from ctypes import CFUNCTYPE, POINTER, c_float, c_int

        def unsafe_caster(result_type):
            assert isinstance(result_type, types.CPointer)

            @intrinsic
            def unsafe_cast(typingctx, src):
                self.assertIsInstance(typingctx, typing.Context)
                if isinstance(src, types.Integer):
                    sig = result_type(types.uintp)

                    def codegen(context, builder, signature, args):
                        [src] = args
                        rtype = signature.return_type
                        llrtype = context.get_value_type(rtype)
                        return builder.inttoptr(src, llrtype)
                    return (sig, codegen)
            return unsafe_cast

        def unsafe_get_ctypes_pointer(src):
            raise NotImplementedError('not callable from python')

        @overload(unsafe_get_ctypes_pointer, strict=False)
        def array_impl_unsafe_get_ctypes_pointer(arrtype):
            if isinstance(arrtype, types.Array):
                unsafe_cast = unsafe_caster(types.CPointer(arrtype.dtype))

                def array_impl(arr):
                    return unsafe_cast(src=arr.ctypes.data)
                return array_impl

        def my_c_fun_raw(ptr, n):
            for i in range(n):
                print(ptr[i])
        prototype = CFUNCTYPE(None, POINTER(c_float), c_int)
        my_c_fun = prototype(my_c_fun_raw)

        @jit(nopython=True)
        def foo(arr):
            ptr = unsafe_get_ctypes_pointer(arr)
            my_c_fun(ptr, arr.size)
        arr = np.arange(10, dtype=np.float32)
        with captured_stdout() as buf:
            foo(arr)
            got = buf.getvalue().splitlines()
        buf.close()
        expect = list(map(str, arr))
        self.assertEqual(expect, got)

    def test_serialization(self):
        """
        Test serialization of intrinsic objects
        """

        @intrinsic
        def identity(context, x):

            def codegen(context, builder, signature, args):
                return args[0]
            sig = x(x)
            return (sig, codegen)

        @jit(nopython=True)
        def foo(x):
            return identity(x)
        self.assertEqual(foo(1), 1)
        memo = _Intrinsic._memo
        memo_size = len(memo)
        serialized_foo = pickle.dumps(foo)
        memo_size += 1
        self.assertEqual(memo_size, len(memo))
        foo_rebuilt = pickle.loads(serialized_foo)
        self.assertEqual(memo_size, len(memo))
        self.assertEqual(foo(1), foo_rebuilt(1))
        serialized_identity = pickle.dumps(identity)
        self.assertEqual(memo_size, len(memo))
        identity_rebuilt = pickle.loads(serialized_identity)
        self.assertIs(identity, identity_rebuilt)
        self.assertEqual(memo_size, len(memo))

    def test_deserialization(self):
        """
        Test deserialization of intrinsic
        """

        def defn(context, x):

            def codegen(context, builder, signature, args):
                return args[0]
            return (x(x), codegen)
        memo = _Intrinsic._memo
        memo_size = len(memo)
        original = _Intrinsic('foo', defn)
        self.assertIs(original._defn, defn)
        pickled = pickle.dumps(original)
        memo_size += 1
        self.assertEqual(memo_size, len(memo))
        del original
        self.assertEqual(memo_size, len(memo))
        _Intrinsic._recent.clear()
        memo_size -= 1
        self.assertEqual(memo_size, len(memo))
        rebuilt = pickle.loads(pickled)
        self.assertIsNot(rebuilt._defn, defn)
        second = pickle.loads(pickled)
        self.assertIs(rebuilt._defn, second._defn)

    def test_docstring(self):

        @intrinsic
        def void_func(typingctx, a: int):
            """void_func docstring"""
            sig = types.void(types.int32)

            def codegen(context, builder, signature, args):
                pass
            return (sig, codegen)
        self.assertEqual('numba.tests.test_extending', void_func.__module__)
        self.assertEqual('void_func', void_func.__name__)
        self.assertEqual('TestIntrinsic.test_docstring.<locals>.void_func', void_func.__qualname__)
        self.assertDictEqual({'a': int}, void_func.__annotations__)
        self.assertEqual('void_func docstring', void_func.__doc__)