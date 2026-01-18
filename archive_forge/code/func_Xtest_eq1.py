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
def Xtest_eq1(self):
    """Various checks for set equality and inequality (1)"""
    try:
        self.assertEqual(self.instance.A == self.instance.tmpset1, True)
        self.assertEqual(self.instance.tmpset1 == self.instance.A, True)
        self.assertEqual(self.instance.A != self.instance.tmpset1, False)
        self.assertEqual(self.instance.tmpset1 != self.instance.A, False)
    except TypeError:
        pass
    else:
        self.fail('fail test_eq1')