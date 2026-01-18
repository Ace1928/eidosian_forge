import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
class TestSetDifference(unittest.TestCase):

    def test_pickle(self):
        a = SetOf([1, 3, 5]) - SetOf([2, 3, 4])
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertEqual(a, b)

    def test_bounds(self):
        a = SetOf([-2, -1, 0, 1])
        b = a - NonNegativeReals
        self.assertEqual(b.bounds(), (-2, -1))
        b = a - RangeSet(3)
        self.assertEqual(b.bounds(), (-2, 0))

    def test_naming(self):
        m = ConcreteModel()
        m.I = SetOf([1, 2])
        a = m.I - [3, 4]
        b = [-1, 1] - a
        self.assertEqual(str(a), 'I - {3, 4}')
        self.assertEqual(str(b), '{-1, 1} - (I - {3, 4})')
        m.A = a
        self.assertEqual(str(a), 'A')
        self.assertEqual(str(b), '{-1, 1} - A')

    def test_domain_and_pprint(self):
        m = ConcreteModel()
        m.I = SetOf([1, 2])
        m.A = m.I - [3, 4]
        self.assertIs(m.A._domain, m.A)
        m.A._domain = Any
        self.assertIs(m.A._domain, m.A)
        with self.assertRaisesRegex(ValueError, 'Setting the domain of a Set Operator is not allowed'):
            m.A._domain = None
        output = StringIO()
        m.A.pprint(ostream=output)
        ref = '\nA : Size=1, Index=None, Ordered=True\n    Key  : Dimen : Domain     : Size : Members\n    None :     1 : I - {3, 4} :    2 : {1, 2}\n'.strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1, 2, 3, 4])
        m.I2 = SetOf([(1, 2), (3, 4)])
        m.IN = SetOf([(1, 2), (3, 4), 1, 2])
        m.J = Set()
        self.assertEqual((m.I1 - m.I1).dimen, 1)
        self.assertEqual((m.I2 - m.I2).dimen, 2)
        self.assertEqual((m.IN - m.IN).dimen, None)
        self.assertEqual((m.I1 - m.I2).dimen, 1)
        self.assertEqual((m.I2 - m.I1).dimen, 2)
        self.assertEqual((m.IN - m.I2).dimen, None)
        self.assertEqual((m.I2 - m.IN).dimen, 2)
        self.assertEqual((m.IN - m.I1).dimen, None)
        self.assertEqual((m.I1 - m.IN).dimen, 1)
        self.assertEqual((m.I1 - m.J).dimen, 1)
        self.assertEqual((m.I2 - m.J).dimen, 2)
        self.assertEqual((m.IN - m.J).dimen, None)
        self.assertEqual((m.J - m.I1).dimen, UnknownSetDimen)
        self.assertEqual((m.J - m.I2).dimen, UnknownSetDimen)
        self.assertEqual((m.J - m.IN).dimen, UnknownSetDimen)

    def _verify_ordered_difference(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.isordered()
        else:
            a_ordered = type(a) is list
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.isordered()
        else:
            b_ordered = type(b) is list
        self.assertTrue(a_ordered)
        x = a - b
        self.assertIs(type(x), SetDifference_OrderedSet)
        self.assertTrue(x.isfinite())
        self.assertTrue(x.isordered())
        self.assertEqual(len(x), 3)
        self.assertEqual(list(x), [3, 2, 5])
        self.assertEqual(x.ordered_data(), (3, 2, 5))
        self.assertEqual(x.sorted_data(), (2, 3, 5))
        self.assertNotIn(0, x)
        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertNotIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)
        self.assertEqual(x.ord(2), 2)
        self.assertEqual(x.ord(3), 1)
        self.assertEqual(x.ord(5), 3)
        with self.assertRaisesRegex(IndexError, 'Cannot identify position of 6 in Set SetDifference_OrderedSet'):
            x.ord(6)
        self.assertEqual(x[1], 3)
        self.assertEqual(x[2], 2)
        self.assertEqual(x[3], 5)
        with self.assertRaisesRegex(IndexError, 'SetDifference_OrderedSet index out of range'):
            x[4]
        self.assertEqual(x[-1], 5)
        self.assertEqual(x[-2], 2)
        self.assertEqual(x[-3], 3)
        with self.assertRaisesRegex(IndexError, 'SetDifference_OrderedSet index out of range'):
            x[-4]

    def test_ordered_setdifference(self):
        self._verify_ordered_difference(SetOf([0, 3, 2, 1, 5, 4]), SetOf([0, 1, 4]))
        self._verify_ordered_difference(SetOf([0, 3, 2, 1, 5, 4]), SetOf({0, 1, 4}))
        self._verify_ordered_difference(SetOf([0, 3, 2, 1, 5, 4]), [0, 1, 4])
        self._verify_ordered_difference(SetOf([0, 3, 2, 1, 5, 4]), {0, 1, 4})
        self._verify_ordered_difference(SetOf([0, 3, 2, 1, 5, 4]), RangeSet(ranges=(NR(0, 1, 0), NR(4, 4, 0))))
        self._verify_ordered_difference([0, 3, 2, 1, 5, 4], SetOf([0, 1, 4]))
        self._verify_ordered_difference([0, 3, 2, 1, 5, 4], SetOf({0, 1, 4}))

    def _verify_finite_difference(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_finite = a.isfinite()
        else:
            a_finite = True
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_finite = b.isfinite()
        else:
            b_finite = True
        self.assertTrue(a_finite or b_finite)
        x = a - b
        self.assertIs(type(x), SetDifference_FiniteSet)
        self.assertTrue(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertEqual(len(x), 3)
        self.assertEqual(sorted(list(x)), [2, 3, 5])
        self.assertEqual(x.ordered_data(), (2, 3, 5))
        self.assertEqual(x.sorted_data(), (2, 3, 5))
        self.assertNotIn(0, x)
        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertNotIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)
        self.assertEqual(len(list(x._sets[0].ranges()) + list(x._sets[1].ranges())), 9)
        self.assertEqual(len(list(x.ranges())), 3)

    def test_finite_setdifference(self):
        self._verify_finite_difference(SetOf({0, 3, 2, 1, 5, 4}), SetOf({0, 1, 4}))
        self._verify_finite_difference(SetOf({0, 3, 2, 1, 5, 4}), SetOf([0, 1, 4]))
        self._verify_finite_difference(SetOf({0, 3, 2, 1, 5, 4}), [0, 1, 4])
        self._verify_finite_difference(SetOf({0, 3, 2, 1, 5, 4}), {0, 1, 4})
        self._verify_finite_difference(SetOf({0, 3, 2, 1, 5, 4}), RangeSet(ranges=(NR(0, 1, 0), NR(4, 4, 0), NR(6, 10, 0))))
        self._verify_finite_difference({0, 3, 2, 1, 5, 4}, SetOf([0, 1, 4]))
        self._verify_finite_difference({0, 3, 2, 1, 5, 4}, SetOf({0, 1, 4}))

    def test_infinite_setdifference(self):
        x = RangeSet(0, 4, 0) - RangeSet(2, 6, 0)
        self.assertIs(type(x), SetDifference_InfiniteSet)
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertNotIn(-1, x)
        self.assertIn(0, x)
        self.assertIn(1, x)
        self.assertIn(1.9, x)
        self.assertNotIn(2, x)
        self.assertNotIn(6, x)
        self.assertNotIn(8, x)
        self.assertEqual(list(x.ranges()), list(RangeSet(ranges=[NR(0, 2, 0, (True, False))]).ranges()))
        x = RangeSet(0, 6, 0) - RangeSet(1, 5, 2)
        self.assertIs(type(x), SetDifference_InfiniteSet)
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertIn(0, x)
        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertNotIn(3, x)
        self.assertIn(4, x)
        self.assertNotIn(5, x)
        self.assertIn(6, x)
        self.assertNotIn(7, x)
        self.assertEqual(list(x.ranges()), list(RangeSet(ranges=[NR(0, 1, 0, (True, False)), NR(1, 3, 0, (False, False)), NR(3, 5, 0, (False, False)), NR(5, 6, 0, (False, True))]).ranges()))