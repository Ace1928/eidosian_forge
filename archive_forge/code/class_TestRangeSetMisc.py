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
class TestRangeSetMisc(unittest.TestCase):

    def test_constructor1(self):
        a = RangeSet(10)
        a.construct()
        tmp = []
        for i in a:
            tmp.append(i)
        self.assertEqual(tmp, list(range(1, 11)))
        self.assertEqual(a.bounds(), (1, 10))

    def test_constructor2(self):
        a = RangeSet(1, 10, 2)
        a.construct()
        tmp = []
        for i in a:
            tmp.append(i)
        self.assertEqual(tmp, list(range(1, 11, 2)))
        self.assertEqual(a.bounds(), (1, 9))

    def test_constructor3(self):
        model = AbstractModel()
        model.a = Param(initialize=1)
        model.b = Param(initialize=2)
        model.c = Param(initialize=10)
        model.d = RangeSet(model.a * model.a, model.c * model.a, model.a * model.b)
        instance = model.create_instance()
        tmp = []
        for i in instance.d:
            tmp.append(i)
        self.assertEqual(tmp, list(range(1, 11, 2)))
        self.assertEqual(instance.d.bounds(), (1, 9))