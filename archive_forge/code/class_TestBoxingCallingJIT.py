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
class TestBoxingCallingJIT(TestCase):

    def setUp(self):
        super().setUp()
        many = base_dummy_type_factory('mydummy2')
        self.DynTypeType, self.DynType, self.dyn_type_type = many
        self.dyn_type = self.DynType()

    def test_unboxer_basic(self):
        magic_token = 51966
        magic_offset = 123

        @intrinsic
        def my_intrinsic(typingctx, val):

            def impl(context, builder, sig, args):
                [val] = args
                return builder.add(val, val.type(magic_offset))
            sig = signature(val, val)
            return (sig, impl)

        @unbox(self.DynTypeType)
        def unboxer(typ, obj, c):

            def bridge(x):
                return my_intrinsic(x)
            args = [c.context.get_constant(types.intp, magic_token)]
            sig = signature(types.voidptr, types.intp)
            is_error, res = c.pyapi.call_jit_code(bridge, sig, args)
            return NativeValue(res, is_error=is_error)

        @box(self.DynTypeType)
        def boxer(typ, val, c):
            res = c.builder.ptrtoint(val, cgutils.intp_t)
            return c.pyapi.long_from_ssize_t(res)

        @njit
        def passthru(x):
            return x
        out = passthru(self.dyn_type)
        self.assertEqual(out, magic_token + magic_offset)

    def test_unboxer_raise(self):

        @unbox(self.DynTypeType)
        def unboxer(typ, obj, c):

            def bridge(x):
                if x > 0:
                    raise ValueError('cannot be x > 0')
                return x
            args = [c.context.get_constant(types.intp, 1)]
            sig = signature(types.voidptr, types.intp)
            is_error, res = c.pyapi.call_jit_code(bridge, sig, args)
            return NativeValue(res, is_error=is_error)

        @box(self.DynTypeType)
        def boxer(typ, val, c):
            res = c.builder.ptrtoint(val, cgutils.intp_t)
            return c.pyapi.long_from_ssize_t(res)

        @njit
        def passthru(x):
            return x
        with self.assertRaises(ValueError) as raises:
            passthru(self.dyn_type)
        self.assertIn('cannot be x > 0', str(raises.exception))

    def test_boxer(self):
        magic_token = 51966
        magic_offset = 312

        @intrinsic
        def my_intrinsic(typingctx, val):

            def impl(context, builder, sig, args):
                [val] = args
                return builder.add(val, val.type(magic_offset))
            sig = signature(val, val)
            return (sig, impl)

        @unbox(self.DynTypeType)
        def unboxer(typ, obj, c):
            return NativeValue(c.context.get_dummy_value())

        @box(self.DynTypeType)
        def boxer(typ, val, c):

            def bridge(x):
                return my_intrinsic(x)
            args = [c.context.get_constant(types.intp, magic_token)]
            sig = signature(types.intp, types.intp)
            is_error, res = c.pyapi.call_jit_code(bridge, sig, args)
            return c.pyapi.long_from_ssize_t(res)

        @njit
        def passthru(x):
            return x
        r = passthru(self.dyn_type)
        self.assertEqual(r, magic_token + magic_offset)

    def test_boxer_raise(self):

        @unbox(self.DynTypeType)
        def unboxer(typ, obj, c):
            return NativeValue(c.context.get_dummy_value())

        @box(self.DynTypeType)
        def boxer(typ, val, c):

            def bridge(x):
                if x > 0:
                    raise ValueError('cannot do x > 0')
                return x
            args = [c.context.get_constant(types.intp, 1)]
            sig = signature(types.intp, types.intp)
            is_error, res = c.pyapi.call_jit_code(bridge, sig, args)
            retval = cgutils.alloca_once(c.builder, c.pyapi.pyobj, zfill=True)
            with c.builder.if_then(c.builder.not_(is_error)):
                obj = c.pyapi.long_from_ssize_t(res)
                c.builder.store(obj, retval)
            return c.builder.load(retval)

        @njit
        def passthru(x):
            return x
        with self.assertRaises(ValueError) as raises:
            passthru(self.dyn_type)
        self.assertIn('cannot do x > 0', str(raises.exception))