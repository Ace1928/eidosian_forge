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
def _verify_ordered_intersection(self, a, b):
    if isinstance(a, (Set, SetOf, RangeSet)):
        a_ordered = a.isordered()
    else:
        a_ordered = type(a) is list
    if isinstance(b, (Set, SetOf, RangeSet)):
        b_ordered = b.isordered()
    else:
        b_ordered = type(b) is list
    self.assertTrue(a_ordered or b_ordered)
    if a_ordered:
        ref = (3, 2, 5)
    else:
        ref = (2, 3, 5)
    x = a & b
    self.assertIs(type(x), SetIntersection_OrderedSet)
    self.assertTrue(x.isfinite())
    self.assertTrue(x.isordered())
    self.assertEqual(len(x), 3)
    self.assertEqual(list(x), list(ref))
    self.assertEqual(x.ordered_data(), tuple(ref))
    self.assertEqual(x.sorted_data(), (2, 3, 5))
    self.assertNotIn(1, x)
    self.assertIn(2, x)
    self.assertIn(3, x)
    self.assertNotIn(4, x)
    self.assertIn(5, x)
    self.assertNotIn(6, x)
    self.assertEqual(x.ord(2), ref.index(2) + 1)
    self.assertEqual(x.ord(3), ref.index(3) + 1)
    self.assertEqual(x.ord(5), 3)
    with self.assertRaisesRegex(IndexError, 'Cannot identify position of 6 in Set SetIntersection_OrderedSet'):
        x.ord(6)
    self.assertEqual(x[1], ref[0])
    self.assertEqual(x[2], ref[1])
    self.assertEqual(x[3], 5)
    with self.assertRaisesRegex(IndexError, 'SetIntersection_OrderedSet index out of range'):
        x[4]
    self.assertEqual(x[-1], 5)
    self.assertEqual(x[-2], ref[-2])
    self.assertEqual(x[-3], ref[-3])
    with self.assertRaisesRegex(IndexError, 'SetIntersection_OrderedSet index out of range'):
        x[-4]