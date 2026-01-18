import re
import unittest
from oslo_config import types
class TypeTestHelper:

    def setUp(self):
        super(TypeTestHelper, self).setUp()
        self.type_instance = self.type

    def assertConvertedValue(self, s, expected):
        self.assertEqual(expected, self.type_instance(s))

    def assertInvalid(self, value):
        self.assertRaises(ValueError, self.type_instance, value)