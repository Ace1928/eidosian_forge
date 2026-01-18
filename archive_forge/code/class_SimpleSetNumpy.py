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
@unittest.skipIf(not _has_numpy, 'Numpy is not installed')
class SimpleSetNumpy(SimpleSetA):

    def setUp(self):
        PyomoModel.setUp(self)
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set A := 1.0 3 5.0 7.0; end;\n')
        OUTPUT.close()
        self.model.A = Set()
        self.model.tmpset1 = Set(initialize=[1.0, 3.0, 5, 7])
        self.model.tmpset2 = Set(initialize=[1.0, 2, 3.0, 5, 7])
        self.model.tmpset3 = Set(initialize=[2, 3.0, 5, 7, 9.1])
        self.model.setunion = Set(initialize=[1.0, 2, 3.0, 5, 7, 9.1])
        self.model.setintersection = Set(initialize=[3.0, 5, 7])
        self.model.setxor = Set(initialize=[1.0, 2, 9.1])
        self.model.setdiff = Set(initialize=[1.0])
        self.model.setmul = Set(initialize=[(1.0, 2), (1.0, 3.0), (1.0, 5), (1.0, 7), (1.0, 9.1), (3.0, 2), (3.0, 3.0), (3.0, 5), (3.0, 7), (3.0, 9.1), (5, 2), (5, 3.0), (5, 5), (5, 7), (5, 9.1), (7, 2), (7, 3.0), (7, 5), (7, 7), (7, 9.1)])
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.e1 = numpy.bool_(1)
        self.e2 = numpy.int_(2)
        self.e3 = numpy.float_(3.0)
        self.e4 = numpy.int_(4)
        self.e5 = numpy.int_(5)
        self.e6 = numpy.int_(6)

    def test_numpy_bool(self):
        model = ConcreteModel()
        model.A = Set(initialize=[numpy.bool_(False), numpy.bool_(True)])
        self.assertEqual(model.A.bounds(), (0, 1))

    def test_numpy_int(self):
        model = ConcreteModel()
        model.A = Set(initialize=[numpy.int_(1.0), numpy.int_(0.0)])
        self.assertEqual(model.A.bounds(), (0, 1))

    def test_numpy_float(self):
        model = ConcreteModel()
        model.A = Set(initialize=[numpy.float_(1.0), numpy.float_(0.0)])
        self.assertEqual(model.A.bounds(), (0, 1))