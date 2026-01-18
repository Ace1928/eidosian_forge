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
def Xtest_clear(self):
    """Check the clear() method empties the set"""
    self.instance.A.clear()
    for key in self.instance.A:
        self.assertEqual(len(self.instance.A[key]), 0)