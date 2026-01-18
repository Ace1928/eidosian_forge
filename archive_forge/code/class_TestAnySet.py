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
class TestAnySet(SimpleSetA):

    def setUp(self):
        PyomoModel.setUp(self)
        x = _AnySet()
        self.model.A = x
        x.concrete = False
        self.model.tmpset1 = Set(initialize=[1, '3', 5, 7])
        self.model.tmpset2 = Set(initialize=[1, 2, '3', 5, 7])
        self.model.tmpset3 = Set(initialize=[2, '3', 5, 7, 9])
        y = _AnySet()
        self.model.setunion = y
        y.concrete = False
        self.model.setintersection = Set(initialize=[1, '3', 5, 7])
        self.model.setxor = Set(initialize=[])
        self.model.setdiff = Set(initialize=[])
        self.model.setmul = None
        self.instance = self.model.create_instance()
        self.e1 = 1
        self.e2 = 2
        self.e3 = '3'
        self.e4 = 4
        self.e5 = 5
        self.e6 = 6

    def test_bounds(self):
        self.assertEqual(self.instance.A.bounds(), (None, None))

    def test_contains(self):
        """Various checks for contains() method"""
        self.assertEqual(self.e1 in self.instance.A, True)
        self.assertEqual(self.e2 in self.instance.A, True)
        self.assertEqual('2' in self.instance.A, True)

    def test_None1(self):
        self.assertEqual(None in Any, True)

    def test_len(self):
        """Check that the set has the right size"""
        with self.assertRaisesRegex(TypeError, "object of type 'Any' has no len()"):
            len(self.instance.A)

    def test_data(self):
        """Check that we can access the underlying set data"""
        with self.assertRaises(AttributeError):
            self.instance.A.data()

    def test_clear(self):
        """Check that the clear() method generates an exception"""
        self.assertIsNone(self.instance.A.clear())

    def test_virtual(self):
        """Check if this is not a virtual set"""
        self.assertEqual(self.instance.A.virtual, True)

    def test_discardValid(self):
        """Check that we fail to remove an invalid set element without an exception"""
        with self.assertRaises(AttributeError):
            self.instance.A.discard(self.e2)

    def test_discardInvalid(self):
        """Check that we fail to remove an invalid set element without an exception"""
        with self.assertRaises(AttributeError):
            self.instance.A.data()

    def test_removeValid(self):
        """Check that we can remove a valid set element"""
        with self.assertRaises(AttributeError):
            self.instance.A.remove(self.e3)

    def test_removeInvalid(self):
        pass

    def test_addInvalid(self):
        """Check that we get an error when adding invalid set elements"""
        pass

    def test_addValid(self):
        """Check that we can add valid set elements"""
        self.assertIs(self.instance.A.domain, Any)
        with self.assertRaises(AttributeError):
            self.instance.A.add(2)

    def test_iterator(self):
        """Check that we can iterate through the set"""
        with self.assertRaises(TypeError):
            for val in self.instance.A:
                pass

    def test_eq1(self):
        """Various checks for set equality and inequality (1)"""
        self.assertTrue(not self.instance.A == self.instance.tmpset1)
        self.assertTrue(not self.instance.tmpset1 == self.instance.A)
        self.assertTrue(self.instance.A != self.instance.tmpset1)
        self.assertTrue(self.instance.tmpset1 != self.instance.A)

    def test_eq2(self):
        """Various checks for set equality and inequality (2)"""
        self.assertTrue(not self.instance.A == self.instance.tmpset2)
        self.assertTrue(not self.instance.tmpset2 == self.instance.A)
        self.assertTrue(self.instance.A != self.instance.tmpset2)
        self.assertTrue(self.instance.tmpset2 != self.instance.A)

    def test_le1(self):
        """Various checks for set subset (1)"""
        self.assertFalse(self.instance.A < self.instance.tmpset1)
        self.assertFalse(self.instance.A <= self.instance.tmpset1)
        self.assertTrue(self.instance.A > self.instance.tmpset1)
        self.assertTrue(self.instance.A >= self.instance.tmpset1)
        self.assertTrue(self.instance.tmpset1 < self.instance.A)
        self.assertTrue(self.instance.tmpset1 <= self.instance.A)
        self.assertFalse(self.instance.tmpset1 > self.instance.A)
        self.assertFalse(self.instance.tmpset1 >= self.instance.A)

    def test_le2(self):
        """Various checks for set subset (2)"""
        self.assertFalse(self.instance.A < self.instance.tmpset2)
        self.assertFalse(self.instance.A <= self.instance.tmpset2)
        self.assertTrue(self.instance.A > self.instance.tmpset2)
        self.assertTrue(self.instance.A >= self.instance.tmpset2)
        self.assertTrue(self.instance.tmpset2 < self.instance.A)
        self.assertTrue(self.instance.tmpset2 <= self.instance.A)
        self.assertFalse(self.instance.tmpset2 > self.instance.A)
        self.assertFalse(self.instance.tmpset2 >= self.instance.A)

    def test_le3(self):
        """Various checks for set subset (3)"""
        self.assertFalse(self.instance.A < self.instance.tmpset3)
        self.assertFalse(self.instance.A <= self.instance.tmpset3)
        self.assertTrue(self.instance.A > self.instance.tmpset3)
        self.assertTrue(self.instance.A >= self.instance.tmpset3)
        self.assertTrue(self.instance.tmpset3 < self.instance.A)
        self.assertTrue(self.instance.tmpset3 <= self.instance.A)
        self.assertFalse(self.instance.tmpset3 > self.instance.A)
        self.assertFalse(self.instance.tmpset3 >= self.instance.A)

    def test_or(self):
        """Check that set union works"""
        self.assertEqual(self.instance.A | self.instance.tmpset3, Any)

    def test_and(self):
        """Check that set intersection works"""
        self.assertEqual(self.instance.A & self.instance.tmpset3, self.instance.tmpset3)

    def test_xor(self):
        """Check that set exclusive or works"""
        self.assertEqual(self.instance.A ^ self.instance.tmpset3, Any)

    def test_diff(self):
        """Check that set difference works"""
        self.assertEqual(self.instance.A - self.instance.tmpset3, Any)

    def test_mul(self):
        """Check that set cross-product works"""
        x = self.instance.A * self.instance.tmpset3
        self.assertIsNone(x.dimen)
        self.assertEqual(list(x.subsets()), [self.instance.A, self.instance.tmpset3])