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
def _verify_infinite_intersection(self, a, b):
    if isinstance(a, (Set, SetOf, RangeSet)):
        a_finite = a.isfinite()
    else:
        a_finite = True
    if isinstance(b, (Set, SetOf, RangeSet)):
        b_finite = b.isfinite()
    else:
        b_finite = True
    self.assertEqual([a_finite, b_finite], [False, False])
    x = a & b
    self.assertIs(type(x), SetIntersection_InfiniteSet)
    self.assertFalse(x.isfinite())
    self.assertFalse(x.isordered())
    self.assertNotIn(1, x)
    self.assertIn(2, x)
    self.assertIn(3, x)
    self.assertIn(4, x)
    self.assertNotIn(5, x)
    self.assertNotIn(6, x)
    self.assertEqual(list(x.ranges()), list(RangeSet(2, 4, 0).ranges()))