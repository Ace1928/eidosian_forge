import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
class ArrayParam6(unittest.TestCase):

    def setUp(self, **kwds):
        self.model = AbstractModel()
        self.repn = '_bogus_'
        self.instance = None

    def tearDown(self):
        self.model = None
        self.instance = None

    def test_index1(self):
        self.model.A = Set(initialize=range(0, 4))

        def B_index(model):
            for i in model.A:
                if i % 2 == 0:
                    yield i

        def B_init(model, i, j):
            if j:
                return 2 + i
            return -(2 + i)
        self.model.B = Param(B_index, [True, False], initialize=B_init)
        self.instance = self.model.create_instance()
        self.assertEqual(set(self.instance.B.keys()), set([(0, True), (2, True), (0, False), (2, False)]))
        self.assertEqual(self.instance.B[0, True], 2)
        self.assertEqual(self.instance.B[0, False], -2)
        self.assertEqual(self.instance.B[2, True], 4)
        self.assertEqual(self.instance.B[2, False], -4)

    def test_index2(self):
        self.model.A = Set(initialize=range(0, 4))

        @set_options(dimen=3)
        def B_index(model):
            return [(i, 2 * i, i * i) for i in model.A if i % 2 == 0]

        def B_init(model, i, ii, iii, j):
            if j:
                return 2 + i
            return -(2 + i)
        self.model.B = Param(B_index, [True, False], initialize=B_init)
        self.instance = self.model.create_instance()
        self.assertEqual(set(self.instance.B.keys()), set([(0, 0, 0, True), (2, 4, 4, True), (0, 0, 0, False), (2, 4, 4, False)]))
        self.assertEqual(self.instance.B[0, 0, 0, True], 2)
        self.assertEqual(self.instance.B[0, 0, 0, False], -2)
        self.assertEqual(self.instance.B[2, 4, 4, True], 4)
        self.assertEqual(self.instance.B[2, 4, 4, False], -4)

    def test_index3(self):
        self.model.A = Set(initialize=range(0, 4))

        def B_index(model):
            return [(i, 2 * i, i * i) for i in model.A if i % 2 == 0]

        def B_init(model, i, ii, iii, j):
            if j:
                return 2 + i
            return -(2 + i)
        self.model.B = Param(B_index, [True, False], initialize=B_init)
        self.instance = self.model.create_instance()
        self.assertEqual(set(self.instance.B.keys()), set([(0, 0, 0, True), (2, 4, 4, True), (0, 0, 0, False), (2, 4, 4, False)]))
        self.assertEqual(self.instance.B[0, 0, 0, True], 2)
        self.assertEqual(self.instance.B[0, 0, 0, False], -2)
        self.assertEqual(self.instance.B[2, 4, 4, True], 4)
        self.assertEqual(self.instance.B[2, 4, 4, False], -4)

    def test_index4(self):
        self.model.A = Set(initialize=range(0, 4))

        @set_options(within=Integers)
        def B_index(model):
            return [i / 2.0 for i in model.A]

        def B_init(model, i, j):
            if j:
                return 2 + i
            return -(2 + i)
        self.model.B = Param(B_index, [True, False], initialize=B_init)
        try:
            self.instance = self.model.create_instance()
            self.fail('Expected ValueError because B_index returns invalid index values')
        except ValueError:
            pass

    def test_dimen1(self):
        model = AbstractModel()
        model.A = Set(dimen=2, initialize=[(1, 2), (3, 4)])
        model.B = Set(dimen=3, initialize=[(1, 1, 1), (2, 2, 2), (3, 3, 3)])
        model.C = Set(dimen=1, initialize=[9, 8, 7, 6, 5])
        model.x = Param(model.A, model.B, model.C, initialize=-1)
        model.y = Param(model.B, initialize=1)
        instance = model.create_instance()
        self.assertEqual(instance.x.dim(), 6)
        self.assertEqual(instance.y.dim(), 3)

    def test_setitem(self):
        model = ConcreteModel()
        model.a = Set(initialize=[1, 2, 3])
        model.b = Set(initialize=['a', 'b', 'c'])
        model.c = model.b * model.b
        model.p = Param(model.a, model.c, within=NonNegativeIntegers, default=0, mutable=True)
        model.p[1, 'a', 'b'] = 1
        model.p[1, ('a', 'b')] = 1
        model.p[(1, 'b'), 'b'] = 1
        try:
            model.p[1, 5, 7] = 1
            self.fail('Expected KeyError')
        except KeyError:
            pass