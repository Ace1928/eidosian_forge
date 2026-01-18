from collections import namedtuple
import gc
import os
import operator
import sys
import weakref
import numpy as np
from numba.core import types, typing, errors, sigutils
from numba.core.types.abstract import _typecache
from numba.core.types.functions import _header_lead
from numba.core.typing.templates import make_overload_template
from numba import jit, njit, typeof
from numba.core.extending import (overload, register_model, models, unbox,
from numba.tests.support import TestCase, create_temp_module
from numba.tests.enum_usecases import Color, Shake, Shape
import unittest
from numba.np import numpy_support
from numba.core import types
class TestNumbers(TestCase):
    """
    Tests for number types.
    """

    def test_bitwidth(self):
        """
        All numeric types have bitwidth attribute
        """
        for ty in types.number_domain:
            self.assertTrue(hasattr(ty, 'bitwidth'))

    def test_minval_maxval(self):
        self.assertEqual(types.int8.maxval, 127)
        self.assertEqual(types.int8.minval, -128)
        self.assertEqual(types.uint8.maxval, 255)
        self.assertEqual(types.uint8.minval, 0)
        self.assertEqual(types.int64.maxval, (1 << 63) - 1)
        self.assertEqual(types.int64.minval, -(1 << 63))
        self.assertEqual(types.uint64.maxval, (1 << 64) - 1)
        self.assertEqual(types.uint64.minval, 0)

    def test_from_bidwidth(self):
        f = types.Integer.from_bitwidth
        self.assertIs(f(32), types.int32)
        self.assertIs(f(8, signed=False), types.uint8)

    def test_ordering(self):

        def check_order(values):
            for i in range(len(values)):
                self.assertLessEqual(values[i], values[i])
                self.assertGreaterEqual(values[i], values[i])
                self.assertFalse(values[i] < values[i])
                self.assertFalse(values[i] > values[i])
                for j in range(i):
                    self.assertLess(values[j], values[i])
                    self.assertLessEqual(values[j], values[i])
                    self.assertGreater(values[i], values[j])
                    self.assertGreaterEqual(values[i], values[j])
                    self.assertFalse(values[i] < values[j])
                    self.assertFalse(values[i] <= values[j])
                    self.assertFalse(values[j] > values[i])
                    self.assertFalse(values[j] >= values[i])
        check_order([types.int8, types.int16, types.int32, types.int64])
        check_order([types.uint8, types.uint16, types.uint32, types.uint64])
        check_order([types.float32, types.float64])
        check_order([types.complex64, types.complex128])
        with self.assertRaises(TypeError):
            types.int8 <= types.uint32
        with self.assertRaises(TypeError):
            types.int8 <= types.float32
        with self.assertRaises(TypeError):
            types.float64 <= types.complex128