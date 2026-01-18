import os, sys, subprocess
import dis
import itertools
import numpy as np
import numba
from numba import jit, njit
from numba.core import errors, ir, types, typing, typeinfer, utils
from numba.core.typeconv import Conversion
from numba.extending import overload_method
from numba.tests.support import TestCase, tag
from numba.tests.test_typeconv import CompatibilityTestMixin
from numba.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.core.typed_passes import PartialTypeInference
from numba.core.compiler_machinery import FunctionPass, register_pass
import unittest
class TestResolveOverload(unittest.TestCase):
    """
    Tests for typing.Context.resolve_overload().
    """

    def assert_resolve_overload(self, cases, args, expected):
        ctx = typing.Context()
        got = ctx.resolve_overload('foo', cases, args, {})
        self.assertEqual(got, expected)

    def test_non_ambiguous_match(self):

        def check(args, expected):
            self.assert_resolve_overload(cases, args, expected)
            self.assert_resolve_overload(cases[::-1], args, expected)
        cases = [i8(i8, i8), i32(i32, i32), f64(f64, f64)]
        check((i8, i8), cases[0])
        check((i32, i32), cases[1])
        check((f64, f64), cases[2])
        check((i8, i16), cases[1])
        check((i32, i8), cases[1])
        check((i32, i8), cases[1])
        check((f32, f32), cases[2])
        check((u32, u32), cases[2])
        check((i64, i64), cases[2])

    def test_ambiguous_match(self):

        def check(args, expected, expected_reverse):
            self.assert_resolve_overload(cases, args, expected)
            self.assert_resolve_overload(cases[::-1], args, expected_reverse)
        cases = [i16(i16, i16), i32(i32, i32), f64(f64, f64)]
        check((i8, i8), cases[0], cases[1])
        check((u16, u16), cases[1], cases[2])
        cases = [i32(i32, i32), f32(f32, f32)]
        check((u32, u32), cases[0], cases[1])

    def test_ambiguous_error(self):
        ctx = typing.Context()
        cases = [i16(i16, i16), i32(i32, i32)]
        with self.assertRaises(TypeError) as raises:
            ctx.resolve_overload('foo', cases, (i8, i8), {}, allow_ambiguous=False)
        self.assertEqual(str(raises.exception).splitlines(), ['Ambiguous overloading for foo (int8, int8):', '(int16, int16) -> int16', '(int32, int32) -> int32'])