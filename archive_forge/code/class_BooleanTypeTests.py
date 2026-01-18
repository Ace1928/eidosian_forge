import re
import unittest
from oslo_config import types
class BooleanTypeTests(TypeTestHelper, unittest.TestCase):
    type = types.Boolean()

    def test_True(self):
        self.assertConvertedValue('True', True)

    def test_yes(self):
        self.assertConvertedValue('yes', True)

    def test_on(self):
        self.assertConvertedValue('on', True)

    def test_1(self):
        self.assertConvertedValue('1', True)

    def test_False(self):
        self.assertConvertedValue('False', False)

    def test_no(self):
        self.assertConvertedValue('no', False)

    def test_off(self):
        self.assertConvertedValue('off', False)

    def test_0(self):
        self.assertConvertedValue('0', False)

    def test_other_values_produce_error(self):
        self.assertInvalid('foo')

    def test_repr(self):
        self.assertEqual('Boolean', repr(types.Boolean()))

    def test_equal(self):
        self.assertEqual(types.Boolean(), types.Boolean())

    def test_not_equal_to_other_class(self):
        self.assertFalse(types.Boolean() == types.String())