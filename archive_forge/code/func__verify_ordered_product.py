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
def _verify_ordered_product(self, a, b):
    if isinstance(a, (Set, SetOf, RangeSet)):
        a_ordered = a.isordered()
    else:
        a_ordered = type(a) is list
    self.assertTrue(a_ordered)
    if isinstance(b, (Set, SetOf, RangeSet)):
        b_ordered = b.isordered()
    else:
        b_ordered = type(b) is list
    self.assertTrue(b_ordered)
    x = a * b
    self.assertIs(type(x), SetProduct_OrderedSet)
    self.assertTrue(x.isfinite())
    self.assertTrue(x.isordered())
    self.assertEqual(len(x), 6)
    self.assertEqual(list(x), [(3, 6), (3, 5), (1, 6), (1, 5), (2, 6), (2, 5)])
    self.assertEqual(x.ordered_data(), ((3, 6), (3, 5), (1, 6), (1, 5), (2, 6), (2, 5)))
    self.assertEqual(x.sorted_data(), ((1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6)))
    self.assertNotIn(1, x)
    self.assertIn((1, 5), x)
    self.assertIn(((1,), 5), x)
    self.assertNotIn((1, 2, 3), x)
    self.assertNotIn((2, 4), x)
    self.assertEqual(x.ord((3, 6)), 1)
    self.assertEqual(x.ord((3, 5)), 2)
    self.assertEqual(x.ord((1, 6)), 3)
    self.assertEqual(x.ord((1, 5)), 4)
    self.assertEqual(x.ord((2, 6)), 5)
    self.assertEqual(x.ord((2, 5)), 6)
    with self.assertRaisesRegex(IndexError, 'Cannot identify position of \\(3, 4\\) in Set SetProduct_OrderedSet'):
        x.ord((3, 4))
    self.assertEqual(x[1], (3, 6))
    self.assertEqual(x[2], (3, 5))
    self.assertEqual(x[3], (1, 6))
    self.assertEqual(x[4], (1, 5))
    self.assertEqual(x[5], (2, 6))
    self.assertEqual(x[6], (2, 5))
    with self.assertRaisesRegex(IndexError, 'SetProduct_OrderedSet index out of range'):
        x[7]
    self.assertEqual(x[-6], (3, 6))
    self.assertEqual(x[-5], (3, 5))
    self.assertEqual(x[-4], (1, 6))
    self.assertEqual(x[-3], (1, 5))
    self.assertEqual(x[-2], (2, 6))
    self.assertEqual(x[-1], (2, 5))
    with self.assertRaisesRegex(IndexError, 'SetProduct_OrderedSet index out of range'):
        x[-7]