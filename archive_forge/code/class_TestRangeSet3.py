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
class TestRangeSet3(PyomoModel):

    def setUp(self):
        PyomoModel.setUp(self)
        self.model.A = RangeSet(1.0, 5.0, 0.8)
        self.model.tmpset1 = Set(initialize=[1, 2, 3, 4, 5])
        self.model.tmpset2 = Set(initialize=[1, 2, 3, 4, 5, 7])
        self.model.tmpset3 = Set(initialize=[2, 3, 5, 7, 9])
        self.model.setunion = Set(initialize=[1, 2, 3, 4, 5, 7, 9])
        self.model.setintersection = Set(initialize=[2, 3, 5])
        self.model.setxor = Set(initialize=[1, 4, 7, 9])
        self.model.setdiff = Set(initialize=[1, 4])
        self.model.setmul = Set(initialize=[(1, 2), (1, 3), (1, 5), (1, 7), (1, 9), (2, 2), (2, 3), (2, 5), (2, 7), (2, 9), (3, 2), (3, 3), (3, 5), (3, 7), (3, 9), (4, 2), (4, 3), (4, 5), (4, 7), (4, 9), (5, 2), (5, 3), (5, 5), (5, 7), (5, 9)])
        self.instance = self.model.create_instance()
        self.e1 = 1
        self.e2 = 2
        self.e3 = 3
        self.e4 = 4
        self.e5 = 5
        self.e6 = 6

    def test_bounds(self):
        """Verify the bounds on this set"""
        self.assertEqual(self.instance.A.bounds(), (1, 5))