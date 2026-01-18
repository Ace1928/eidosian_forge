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
def _verify_finite_union(self, a, b):
    if isinstance(a, SetOf):
        if type(a._ref) is list:
            self.assertTrue(a.isordered())
        else:
            self.assertFalse(a.isordered())
        self.assertTrue(a.isfinite())
    else:
        self.assertIn(type(a), (list, set))
    if isinstance(b, SetOf):
        if type(b._ref) is list:
            self.assertTrue(b.isordered())
        else:
            self.assertFalse(b.isordered())
        self.assertTrue(b.isfinite())
    else:
        self.assertIn(type(b), (list, set))
    x = a | b
    self.assertIs(type(x), SetUnion_FiniteSet)
    self.assertTrue(x.isfinite())
    self.assertFalse(x.isordered())
    self.assertEqual(len(x), 5)
    if x._sets[0].isordered():
        self.assertEqual(list(x)[:3], [1, 3, 2])
    if x._sets[1].isordered():
        self.assertEqual(list(x)[-2:], [5, 4])
    self.assertEqual(sorted(list(x)), [1, 2, 3, 4, 5])
    self.assertEqual(x.ordered_data(), (1, 2, 3, 4, 5))
    self.assertEqual(x.sorted_data(), (1, 2, 3, 4, 5))
    self.assertIn(1, x)
    self.assertIn(2, x)
    self.assertIn(3, x)
    self.assertIn(4, x)
    self.assertIn(5, x)
    self.assertNotIn(6, x)
    self.assertEqual(len(list(x._sets[0].ranges()) + list(x._sets[1].ranges())), 6)
    self.assertEqual(len(list(x.ranges())), 5)