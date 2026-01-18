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
def Xtest_mul(self):
    """Check that set cross-product works"""
    try:
        self.instance.A * self.instance.tmpset3
    except TypeError:
        pass
    else:
        self.fail('fail test_mul')