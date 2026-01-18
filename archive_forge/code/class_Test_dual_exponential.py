import pickle
import math
import pyomo.common.unittest as unittest
from pyomo.kernel import pprint, IntegerSet
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.variable import variable, variable_tuple
from pyomo.core.kernel.block import block
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression, data_expression
from pyomo.core.kernel.conic import (
class Test_dual_exponential(_conic_tester_base, unittest.TestCase):
    _object_factory = lambda self: dual_exponential(r=variable(lb=0), x1=variable(), x2=variable(ub=0))

    def test_expression(self):
        c = self._object_factory()
        self.assertIs(c._body, None)
        with self.assertRaises(ValueError):
            self.assertIs(c(), None)
        with self.assertRaises(ValueError):
            self.assertIs(c(exception=True), None)
        self.assertIs(c(exception=False), None)
        self.assertIs(c._body, None)
        self.assertIs(c.slack, None)
        self.assertIs(c.lslack, None)
        self.assertIs(c.uslack, None)
        c.r.value = 2.7
        c.x1.value = 1.2
        c.x2.value = -5.3
        val = round(-(-5.3 / math.e) * math.exp(1.2 / -5.3) - 2.7, 9)
        self.assertEqual(round(c(), 9), val)
        self.assertEqual(round(c.slack, 9), -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(round(c.uslack, 9), -val)
        self.assertIs(c._body, None)
        self.assertEqual(round(c.body(), 9), val)
        self.assertEqual(round(c(), 9), val)
        self.assertEqual(round(c.slack, 9), -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(round(c.uslack, 9), -val)
        self.assertIsNot(c._body, None)

    def test_check_convexity_conditions(self):
        c = self._object_factory()
        self.assertEqual(c.check_convexity_conditions(), True)
        c = self._object_factory()
        c.r.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(), False)
        self.assertEqual(c.check_convexity_conditions(relax=True), True)
        c = self._object_factory()
        c.r.lb = None
        self.assertEqual(c.check_convexity_conditions(), False)
        c = self._object_factory()
        c.r.lb = -1
        self.assertEqual(c.check_convexity_conditions(), False)
        c = self._object_factory()
        c.x1.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(), False)
        self.assertEqual(c.check_convexity_conditions(relax=True), True)
        c = self._object_factory()
        c.x2.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(), False)
        self.assertEqual(c.check_convexity_conditions(relax=True), True)
        c = self._object_factory()
        c.x2.ub = None
        self.assertEqual(c.check_convexity_conditions(), False)
        c = self._object_factory()
        c.x2.ub = 1
        self.assertEqual(c.check_convexity_conditions(), False)

    def test_as_domain(self):
        ret = dual_exponential.as_domain(r=3, x1=1, x2=2)
        self.assertIs(type(ret), block)
        q, c, r, x1, x2 = (ret.q, ret.c, ret.r, ret.x1, ret.x2)
        self.assertEqual(q.check_convexity_conditions(), True)
        self.assertIs(type(q), dual_exponential)
        self.assertIs(type(x1), variable)
        self.assertIs(type(x2), variable)
        self.assertIs(type(r), variable)
        self.assertIs(type(c), constraint_tuple)
        self.assertEqual(len(c), 3)
        self.assertEqual(c[0].rhs, 3)
        r.value = 3
        self.assertEqual(c[0].slack, 0)
        r.value = None
        self.assertEqual(c[1].rhs, 1)
        x1.value = 1
        self.assertEqual(c[1].slack, 0)
        x1.value = None
        self.assertEqual(c[2].rhs, 2)
        x2.value = 2
        self.assertEqual(c[2].slack, 0)
        x2.value = None