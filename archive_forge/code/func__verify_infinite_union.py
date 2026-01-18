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
def _verify_infinite_union(self, a, b):
    if isinstance(a, RangeSet):
        self.assertFalse(a.isordered())
        self.assertFalse(a.isfinite())
    else:
        self.assertIn(type(a), (list, set))
    if isinstance(b, RangeSet):
        self.assertFalse(b.isordered())
        self.assertFalse(b.isfinite())
    else:
        self.assertIn(type(b), (list, set))
    x = a | b
    self.assertIs(type(x), SetUnion_InfiniteSet)
    self.assertFalse(x.isfinite())
    self.assertFalse(x.isordered())
    self.assertIn(1, x)
    self.assertIn(2, x)
    self.assertIn(3, x)
    self.assertIn(4, x)
    self.assertIn(5, x)
    self.assertNotIn(6, x)
    self.assertEqual(list(x.ranges()), list(x._sets[0].ranges()) + list(x._sets[1].ranges()))