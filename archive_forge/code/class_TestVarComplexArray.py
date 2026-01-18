import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
from pyomo.core.base.units_container import units, pint_available, UnitsError
class TestVarComplexArray(PyomoModel):

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
        self.model.B = Var(B_index, [True, False], initialize=B_init, dense=True)
        self.instance = self.model.create_instance()
        self.assertEqual(set(self.instance.B.keys()), set([(0, True), (2, True), (0, False), (2, False)]))
        self.assertEqual(self.instance.B[0, True].value, 2)
        self.assertEqual(self.instance.B[0, False].value, -2)
        self.assertEqual(self.instance.B[2, True].value, 4)
        self.assertEqual(self.instance.B[2, False].value, -4)

    def test_index2(self):
        self.model.A = Set(initialize=range(0, 4))

        def B_index(model):
            for i in model.A:
                if i % 2 == 0:
                    yield (i - 1, i)
        B_index.dimen = 2

        def B_init(model, k, i, j):
            if j:
                return (2 + i) * k
            return -(2 + i) * k
        self.model.B = Var(B_index, [True, False], initialize=B_init, dense=True)
        self.instance = self.model.create_instance()
        self.assertEqual(set(self.instance.B.keys()), set([(-1, 0, True), (1, 2, True), (-1, 0, False), (1, 2, False)]))
        self.assertEqual(self.instance.B[-1, 0, True].value, -2)
        self.assertEqual(self.instance.B[-1, 0, False].value, 2)
        self.assertEqual(self.instance.B[1, 2, True].value, 4)
        self.assertEqual(self.instance.B[1, 2, False].value, -4)