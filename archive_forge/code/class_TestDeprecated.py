import unittest
from traits.testing.api import UnittestTools
from traits.util.api import deprecated
class TestDeprecated(unittest.TestCase, UnittestTools):

    def test_deprecated_function(self):
        with self.assertDeprecated():
            result = my_deprecated_addition(42, 1729)
        self.assertEqual(result, 1771)

    def test_deprecated_exception_raising_function(self):
        with self.assertRaises(ZeroDivisionError):
            with self.assertDeprecated():
                my_bad_function()

    def test_deprecated_method(self):
        obj = ClassWithDeprecatedBits()
        with self.assertDeprecated():
            result = obj.bits()
        self.assertEqual(result, 42)

    def test_deprecated_method_with_fancy_signature(self):
        obj = ClassWithDeprecatedBits()
        with self.assertDeprecated():
            result = obj.bytes(3, 27, 65, name='Boris', age=-3.2)
        self.assertEqual(result, (3, (27, 65), {'name': 'Boris', 'age': -3.2}))