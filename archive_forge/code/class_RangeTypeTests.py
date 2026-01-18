import re
import unittest
from oslo_config import types
class RangeTypeTests(TypeTestHelper, unittest.TestCase):
    type = types.Range()

    def assertRange(self, s, r1, r2, step=1):
        self.assertEqual(list(range(r1, r2, step)), list(self.type_instance(s)))

    def test_range(self):
        self.assertRange('0-2', 0, 3)
        self.assertRange('-2-0', -2, 1)
        self.assertRange('2-0', 2, -1, -1)
        self.assertRange('-3--1', -3, 0)
        self.assertRange('-1--3', -1, -4, -1)
        self.assertRange('-1', -1, 0)
        self.assertInvalid('--1')
        self.assertInvalid('4-')
        self.assertInvalid('--')
        self.assertInvalid('1.1-1.2')
        self.assertInvalid('a-b')

    def test_range_bounds(self):
        self.type_instance = types.Range(1, 3)
        self.assertRange('1-3', 1, 4)
        self.assertRange('2-2', 2, 3)
        self.assertRange('2', 2, 3)
        self.assertInvalid('1-4')
        self.assertInvalid('0-3')
        self.assertInvalid('0-4')

    def test_range_exclusive(self):
        self.type_instance = types.Range(inclusive=False)
        self.assertRange('0-2', 0, 2)
        self.assertRange('-2-0', -2, 0)
        self.assertRange('2-0', 2, 0, -1)
        self.assertRange('-3--1', -3, -1)
        self.assertRange('-1--3', -1, -3, -1)
        self.assertRange('-1', -1, -1)