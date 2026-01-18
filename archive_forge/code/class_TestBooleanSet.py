import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
class TestBooleanSet(unittest.TestCase):

    def test_bounds(self):
        x = BooleanSet()
        self.assertEqual(x.bounds(), (0, 1))

    def test_inequality_comparison_fails(self):
        x = RealSet()
        y = RealSet()
        self.assertFalse(x < y)
        self.assertTrue(x <= y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)

    def test_name(self):
        x = BooleanSet()
        self.assertEqual(x.name, 'Boolean')
        self.assertEqual('Boolean', str(x))
        x = BooleanSet(name='x')
        self.assertEqual(x.name, 'x')
        self.assertEqual(str(x), 'x')

    def test_contains(self):
        x = BooleanSet()
        self.assertFalse(None in x)
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(True in x)
        self.assertTrue(1.0 in x)
        self.assertFalse(0.3 in x)
        self.assertTrue(0 in x)
        self.assertTrue(False in x)
        self.assertTrue(0.0 in x)
        self.assertFalse(-0.45 in x)
        self.assertFalse(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)

    def test_Boolean(self):
        x = Boolean
        self.assertFalse(None in x)
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(True in x)
        self.assertTrue(1.0 in x)
        self.assertFalse(0.3 in x)
        self.assertTrue(0 in x)
        self.assertTrue(False in x)
        self.assertTrue(0.0 in x)
        self.assertFalse(-0.45 in x)
        self.assertFalse(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)

    def test_Binary(self):
        x = Binary
        self.assertFalse(None in x)
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(True in x)
        self.assertTrue(1.0 in x)
        self.assertFalse(0.3 in x)
        self.assertTrue(0 in x)
        self.assertTrue(False in x)
        self.assertTrue(0.0 in x)
        self.assertFalse(-0.45 in x)
        self.assertFalse(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)