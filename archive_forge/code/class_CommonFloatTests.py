import unittest
from traits.api import BaseFloat, Either, Float, HasTraits, Str, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
class CommonFloatTests(object):
    """ Common tests for Float and BaseFloat """

    def test_default(self):
        a = self.test_class()
        self.assertEqual(a.value, 0.0)

    def test_accepts_float(self):
        a = self.test_class()
        a.value = 5.6
        self.assertIs(type(a.value), float)
        self.assertEqual(a.value, 5.6)
        a.value_or_none = 5.6
        self.assertIs(type(a.value_or_none), float)
        self.assertEqual(a.value_or_none, 5.6)

    def test_accepts_float_subclass(self):
        a = self.test_class()
        a.value = InheritsFromFloat(37.0)
        self.assertIs(type(a.value), float)
        self.assertEqual(a.value, 37.0)
        a.value_or_none = InheritsFromFloat(37.0)
        self.assertIs(type(a.value), float)
        self.assertEqual(a.value, 37.0)

    def test_accepts_int(self):
        a = self.test_class()
        a.value = 2
        self.assertIs(type(a.value), float)
        self.assertEqual(a.value, 2.0)
        a.value_or_none = 2
        self.assertIs(type(a.value_or_none), float)
        self.assertEqual(a.value_or_none, 2.0)

    def test_accepts_float_like(self):
        a = self.test_class()
        a.value = MyFloat(1729.0)
        self.assertIs(type(a.value), float)
        self.assertEqual(a.value, 1729.0)
        a.value = MyFloat(594.0)
        self.assertIs(type(a.value), float)
        self.assertEqual(a.value, 594.0)

    def test_rejects_string(self):
        a = self.test_class()
        with self.assertRaises(TraitError):
            a.value = '2.3'
        with self.assertRaises(TraitError):
            a.value_or_none = '2.3'

    def test_bad_float_exceptions_propagated(self):
        a = self.test_class()
        with self.assertRaises(ZeroDivisionError):
            a.value = BadFloat()

    def test_compound_trait_float_conversion_fail(self):
        a = self.test_class()
        a.float_or_text = 'not a float'
        self.assertEqual(a.float_or_text, 'not a float')

    def test_accepts_small_integer(self):
        a = self.test_class()
        a.value = 2
        self.assertIs(type(a.value), float)
        self.assertEqual(a.value, 2.0)
        a.value_or_none = 2
        self.assertIs(type(a.value_or_none), float)
        self.assertEqual(a.value_or_none, 2.0)

    def test_accepts_large_integer(self):
        a = self.test_class()
        a.value = 2 ** 64
        self.assertIs(type(a.value), float)
        self.assertEqual(a.value, 2 ** 64)
        a.value_or_none = 2 ** 64
        self.assertIs(type(a.value_or_none), float)
        self.assertEqual(a.value_or_none, 2 ** 64)

    @requires_numpy
    def test_accepts_numpy_floats(self):
        test_values = [numpy.float64(2.3), numpy.float32(3.7), numpy.float16(1.28)]
        a = self.test_class()
        for test_value in test_values:
            a.value = test_value
            self.assertIs(type(a.value), float)
            self.assertEqual(a.value, test_value)
            a.value_or_none = test_value
            self.assertIs(type(a.value_or_none), float)
            self.assertEqual(a.value_or_none, test_value)