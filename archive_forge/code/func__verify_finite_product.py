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
def _verify_finite_product(self, a, b):
    if isinstance(a, (Set, SetOf, RangeSet)):
        a_ordered = a.isordered()
    else:
        a_ordered = type(a) is list
    if isinstance(b, (Set, SetOf, RangeSet)):
        b_ordered = b.isordered()
    else:
        b_ordered = type(b) is list
    self.assertFalse(a_ordered and b_ordered)
    x = a * b
    self.assertIs(type(x), SetProduct_FiniteSet)
    self.assertTrue(x.isfinite())
    self.assertFalse(x.isordered())
    self.assertEqual(len(x), 6)
    self.assertEqual(sorted(list(x)), [(1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6)])
    self.assertEqual(x.ordered_data(), ((1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6)))
    self.assertEqual(x.sorted_data(), ((1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6)))
    self.assertNotIn(1, x)
    self.assertIn((1, 5), x)
    self.assertIn(((1,), 5), x)
    self.assertNotIn((1, 2, 3), x)
    self.assertNotIn((2, 4), x)