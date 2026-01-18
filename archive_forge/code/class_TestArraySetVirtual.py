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
class TestArraySetVirtual(unittest.TestCase):

    def test_construct(self):
        a = Set(initialize=[1, 2, 3])
        a.construct()
        b = Set(a, initialize=virt_constructor)