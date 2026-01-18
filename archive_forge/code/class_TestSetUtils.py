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
class TestSetUtils(unittest.TestCase):

    def test_get_continuous_interval(self):
        self.assertEqual(Reals.get_interval(), (None, None, 0))
        self.assertEqual(PositiveReals.get_interval(), (0, None, 0))
        self.assertEqual(NonNegativeReals.get_interval(), (0, None, 0))
        self.assertEqual(NonPositiveReals.get_interval(), (None, 0, 0))
        self.assertEqual(NegativeReals.get_interval(), (None, 0, 0))
        a = NonNegativeReals | NonPositiveReals
        self.assertEqual(a.get_interval(), (None, None, 0))
        a = NonPositiveReals | NonNegativeReals
        self.assertEqual(a.get_interval(), (None, None, 0))
        a = NegativeReals | PositiveReals
        self.assertEqual(a.get_interval(), (None, None, None))
        a = NegativeReals | PositiveReals | [0]
        self.assertEqual(a.get_interval(), (None, None, 0))
        a = NegativeReals | PositiveReals | RangeSet(0, 5)
        self.assertEqual(a.get_interval(), (None, None, 0))
        a = NegativeReals | RangeSet(-3, 3)
        self.assertEqual(a.get_interval(), (None, 3, None))
        a = NegativeReals | Binary
        self.assertEqual(a.get_interval(), (None, 1, None))
        a = PositiveReals | Binary
        self.assertEqual(a.get_interval(), (0, None, 0))
        a = RangeSet(1, 10, 0) | RangeSet(5, 15, 0)
        self.assertEqual(a.get_interval(), (1, 15, 0))
        a = RangeSet(5, 15, 0) | RangeSet(1, 10, 0)
        self.assertEqual(a.get_interval(), (1, 15, 0))
        a = RangeSet(5, 15, 0) | RangeSet(1, 4, 0)
        self.assertEqual(a.get_interval(), (1, 15, None))
        a = RangeSet(1, 4, 0) | RangeSet(5, 15, 0)
        self.assertEqual(a.get_interval(), (1, 15, None))
        a = NegativeReals | Any
        self.assertEqual(a.get_interval(), (None, None, None))
        a = Any | NegativeReals
        self.assertEqual(a.get_interval(), (None, None, None))
        a = SetOf('abc') | NegativeReals
        self.assertEqual(a.get_interval(), (None, None, None))
        a = NegativeReals | SetOf('abc')
        self.assertEqual(a.get_interval(), (None, None, None))

    def test_get_discrete_interval(self):
        self.assertEqual(Integers.get_interval(), (None, None, 1))
        self.assertEqual(PositiveIntegers.get_interval(), (1, None, 1))
        self.assertEqual(NegativeIntegers.get_interval(), (None, -1, 1))
        self.assertEqual(Binary.get_interval(), (0, 1, 1))
        a = PositiveIntegers | NegativeIntegers
        self.assertEqual(a.get_interval(), (None, None, None))
        a = NegativeIntegers | NonNegativeIntegers
        self.assertEqual(a.get_interval(), (None, None, 1))
        a = SetOf([1, 3, 5, 6, 4, 2])
        self.assertEqual(a.get_interval(), (1, 6, 1))
        a = SetOf([1, 3, 5, 6, 2])
        self.assertEqual(a.get_interval(), (1, 6, None))
        a = SetOf([1, 3, 5, 6, 4, 2, 'a'])
        self.assertEqual(a.get_interval(), (None, None, None))
        a = SetOf([3])
        self.assertEqual(a.get_interval(), (3, 3, 0))
        a = RangeSet(ranges=(NR(0, 5, 1), NR(5, 10, 1)))
        self.assertEqual(a.get_interval(), (0, 10, 1))
        a = RangeSet(ranges=(NR(5, 10, 1), NR(0, 5, 1)))
        self.assertEqual(a.get_interval(), (0, 10, 1))
        a = RangeSet(ranges=(NR(0, 4, 1), NR(5, 10, 1)))
        self.assertEqual(a.get_interval(), (0, 10, 1))
        a = RangeSet(ranges=(NR(5, 10, 1), NR(0, 4, 1)))
        self.assertEqual(a.get_interval(), (0, 10, 1))
        a = RangeSet(ranges=(NR(0, 3, 1), NR(5, 10, 1)))
        self.assertEqual(a.get_interval(), (0, 10, None))
        a = RangeSet(ranges=(NR(5, 10, 1), NR(0, 3, 1)))
        self.assertEqual(a.get_interval(), (0, 10, None))
        a = RangeSet(ranges=(NR(0, 4, 2), NR(6, 10, 2)))
        self.assertEqual(a.get_interval(), (0, 10, 2))
        a = RangeSet(ranges=(NR(6, 10, 2), NR(0, 4, 2)))
        self.assertEqual(a.get_interval(), (0, 10, 2))
        a = RangeSet(ranges=(NR(0, 4, 2), NR(5, 10, 2)))
        self.assertEqual(a.get_interval(), (0, 9, None))
        a = RangeSet(ranges=(NR(5, 10, 2), NR(0, 4, 2)))
        self.assertEqual(a.get_interval(), (0, 9, None))
        a = RangeSet(ranges=(NR(0, 10, 2), NR(0, 10, 3)))
        self.assertEqual(a.get_interval(), (0, 10, None))
        a = RangeSet(ranges=(NR(0, 10, 3), NR(0, 10, 2)))
        self.assertEqual(a.get_interval(), (0, 10, None))
        a = RangeSet(ranges=(NR(2, 10, 2), NR(0, 12, 4)))
        self.assertEqual(a.get_interval(), (0, 12, 2))
        a = RangeSet(ranges=(NR(0, 12, 4), NR(2, 10, 2)))
        self.assertEqual(a.get_interval(), (0, 12, 2))
        a = RangeSet(ranges=(NR(0, 10, 2), NR(1, 10, 2)))
        self.assertEqual(a.get_interval(), (0, 10, None))
        a = RangeSet(ranges=(NR(0, 10, 3), NR(1, 10, 3), NR(2, 10, 3)))
        self.assertEqual(a.get_interval(), (0, 10, None))

    def test_get_interval(self):
        self.assertEqual(Any.get_interval(), (None, None, None))
        a = UnindexedComponent_set
        self.assertEqual(a.get_interval(), (None, None, None))
        a = Set(initialize=['a'])
        a.construct()
        self.assertEqual(a.get_interval(), ('a', 'a', None))
        a = Set(initialize=[1])
        a.construct()
        self.assertEqual(a.get_interval(), (1, 1, 0))