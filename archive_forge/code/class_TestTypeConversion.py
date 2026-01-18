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
class TestTypeConversion(CompatibilityTestMixin, unittest.TestCase):
    """
    Test for conversion between types with a typing context.
    """

    def assert_can_convert(self, aty, bty, expected):
        ctx = typing.Context()
        got = ctx.can_convert(aty, bty)
        self.assertEqual(got, expected)

    def assert_cannot_convert(self, aty, bty):
        ctx = typing.Context()
        got = ctx.can_convert(aty, bty)
        self.assertIsNone(got)

    def test_convert_number_types(self):
        ctx = typing.Context()
        self.check_number_compatibility(ctx.can_convert)

    def test_tuple(self):
        aty = types.UniTuple(i32, 3)
        bty = types.UniTuple(i64, 3)
        self.assert_can_convert(aty, aty, Conversion.exact)
        self.assert_can_convert(aty, bty, Conversion.promote)
        aty = types.UniTuple(i32, 3)
        bty = types.UniTuple(f64, 3)
        self.assert_can_convert(aty, bty, Conversion.safe)
        aty = types.Tuple((i32, i32))
        bty = types.Tuple((i32, i64))
        self.assert_can_convert(aty, bty, Conversion.promote)
        aty = types.UniTuple(i32, 2)
        bty = types.Tuple((i32, i64))
        self.assert_can_convert(aty, bty, Conversion.promote)
        self.assert_can_convert(bty, aty, Conversion.unsafe)
        aty = types.UniTuple(i64, 0)
        bty = types.UniTuple(i32, 0)
        cty = types.Tuple(())
        self.assert_can_convert(aty, bty, Conversion.safe)
        self.assert_can_convert(bty, aty, Conversion.safe)
        self.assert_can_convert(aty, cty, Conversion.safe)
        self.assert_can_convert(cty, aty, Conversion.safe)
        aty = types.UniTuple(i64, 3)
        bty = types.UniTuple(types.none, 3)
        self.assert_cannot_convert(aty, bty)
        aty = types.UniTuple(i64, 2)
        bty = types.UniTuple(i64, 3)

    def test_arrays(self):
        aty = types.Array(i32, 3, 'C')
        bty = types.Array(i32, 3, 'A')
        self.assert_can_convert(aty, bty, Conversion.safe)
        aty = types.Array(i32, 2, 'C')
        bty = types.Array(i32, 2, 'F')
        self.assert_cannot_convert(aty, bty)
        aty = types.Array(i32, 3, 'C')
        bty = types.Array(i32, 3, 'C', readonly=True)
        self.assert_can_convert(aty, aty, Conversion.exact)
        self.assert_can_convert(bty, bty, Conversion.exact)
        self.assert_can_convert(aty, bty, Conversion.safe)
        self.assert_cannot_convert(bty, aty)
        aty = types.Array(i32, 2, 'C')
        bty = types.Array(i32, 3, 'C')
        self.assert_cannot_convert(aty, bty)
        aty = types.Array(i32, 2, 'C')
        bty = types.Array(i64, 2, 'C')
        self.assert_cannot_convert(aty, bty)

    def test_optional(self):
        aty = types.int32
        bty = types.Optional(i32)
        self.assert_can_convert(types.none, bty, Conversion.promote)
        self.assert_can_convert(aty, bty, Conversion.promote)
        self.assert_cannot_convert(bty, types.none)
        self.assert_can_convert(bty, aty, Conversion.safe)
        aty = types.Array(i32, 2, 'C')
        bty = types.Optional(aty)
        self.assert_can_convert(types.none, bty, Conversion.promote)
        self.assert_can_convert(aty, bty, Conversion.promote)
        self.assert_can_convert(bty, aty, Conversion.safe)
        aty = types.Array(i32, 2, 'C')
        bty = types.Optional(aty.copy(layout='A'))
        self.assert_can_convert(aty, bty, Conversion.safe)
        self.assert_cannot_convert(bty, aty)
        aty = types.Array(i32, 2, 'C')
        bty = types.Optional(aty.copy(layout='F'))
        self.assert_cannot_convert(aty, bty)
        self.assert_cannot_convert(bty, aty)