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
class TestRealSet(unittest.TestCase):

    def test_bounds(self):
        x = RealSet()
        self.assertEqual(x.bounds(), (None, None))
        x = RealSet(bounds=(1, 2))
        self.assertEqual(x.bounds(), (1, 2))

    def test_inequality_comparison_fails(self):
        x = RealSet()
        y = RealSet()
        self.assertFalse(x < y)
        self.assertTrue(x <= y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)

    def test_name(self):
        x = RealSet()
        self.assertEqual(x.name, 'Reals')
        self.assertEqual('Reals', str(x))
        x = RealSet(name='x')
        self.assertEqual(x.name, 'x')
        self.assertEqual(str(x), 'x')

    @unittest.skip('_VirtualSet was removed during the set rewrite')
    def test_contains(self):
        x = _VirtualSet()
        self.assertTrue(None in x)
        self.assertTrue(10 in x)
        self.assertTrue(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(0.3 in x)
        self.assertTrue(0 in x)
        self.assertTrue(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertTrue(-2.2 in x)
        self.assertTrue(-10 in x)
        x = RealSet()
        self.assertFalse(None in x)
        self.assertTrue(10 in x)
        self.assertTrue(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(0.3 in x)
        self.assertTrue(0 in x)
        self.assertTrue(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertTrue(-2.2 in x)
        self.assertTrue(-10 in x)
        x = RealSet(bounds=(-1, 1))
        self.assertFalse(None in x)
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(0.3 in x)
        self.assertTrue(0 in x)
        self.assertTrue(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)

    def test_PositiveReals(self):
        x = PositiveReals
        self.assertFalse(None in x)
        self.assertTrue(10 in x)
        self.assertTrue(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(0.3 in x)
        self.assertFalse(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertFalse(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)

    def test_NonPositiveReals(self):
        x = NonPositiveReals
        self.assertFalse(None in x)
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertFalse(1 in x)
        self.assertFalse(0.3 in x)
        self.assertTrue(0 in x)
        self.assertTrue(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertTrue(-2.2 in x)
        self.assertTrue(-10 in x)

    def test_NegativeReals(self):
        x = NegativeReals
        self.assertFalse(None in x)
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertFalse(1 in x)
        self.assertFalse(0.3 in x)
        self.assertFalse(0 in x)
        self.assertTrue(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertTrue(-2.2 in x)
        self.assertTrue(-10 in x)

    def test_NonNegativeReals(self):
        x = NonNegativeReals
        self.assertFalse(None in x)
        self.assertTrue(10 in x)
        self.assertTrue(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(0.3 in x)
        self.assertTrue(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertFalse(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)

    def test_PercentFraction(self):
        x = PercentFraction
        self.assertFalse(None in x)
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(0.3 in x)
        self.assertTrue(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertFalse(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)

    def test_UnitInterval(self):
        x = UnitInterval
        self.assertFalse(None in x)
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(0.3 in x)
        self.assertTrue(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertFalse(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)

    def test_RealInterval(self):
        x = RealInterval()
        self.assertEqual(x.name, "'RealInterval(None, None)'")
        self.assertEqual(x.local_name, 'RealInterval(None, None)')
        self.assertFalse(None in x)
        self.assertTrue(10 in x)
        self.assertTrue(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(0.3 in x)
        self.assertTrue(0 in x)
        self.assertTrue(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertTrue(-2.2 in x)
        self.assertTrue(-10 in x)
        x = RealInterval(bounds=(-1, 1))
        self.assertEqual(x.name, "'RealInterval(-1, 1)'")
        self.assertEqual(x.local_name, 'RealInterval(-1, 1)')
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(0.3 in x)
        self.assertTrue(0 in x)
        self.assertTrue(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)
        x = RealInterval(bounds=(-1, 1), name='JUNK')
        self.assertEqual(x.name, 'JUNK')
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertTrue(0.3 in x)
        self.assertTrue(0 in x)
        self.assertTrue(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)