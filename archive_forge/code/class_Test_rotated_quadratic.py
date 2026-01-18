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
class Test_rotated_quadratic(_conic_tester_base, unittest.TestCase):
    _object_factory = lambda self: rotated_quadratic(r1=variable(lb=0), r2=variable(lb=0), x=[variable(), variable()])

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
        c.r1.value = 5
        c.r2.value = 7
        c.x[0].value = 2
        c.x[1].value = 3
        val = 2 ** 2 + 3 ** 2 - 2 * 5 * 7
        self.assertEqual(c(), val)
        self.assertEqual(c.slack, -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(c.uslack, -val)
        self.assertIs(c._body, None)
        self.assertEqual(c.body(), val)
        self.assertEqual(c(), val)
        self.assertEqual(c.slack, -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(c.uslack, -val)
        self.assertIsNot(c._body, None)

    def test_check_convexity_conditions(self):
        c = self._object_factory()
        self.assertEqual(c.check_convexity_conditions(), True)
        c = self._object_factory()
        c.r1.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(), False)
        self.assertEqual(c.check_convexity_conditions(relax=True), True)
        c = self._object_factory()
        c.r1.lb = None
        self.assertEqual(c.check_convexity_conditions(), False)
        c = self._object_factory()
        c.r1.lb = -1
        self.assertEqual(c.check_convexity_conditions(), False)
        c = self._object_factory()
        c.r2.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(), False)
        self.assertEqual(c.check_convexity_conditions(relax=True), True)
        c = self._object_factory()
        c.r2.lb = None
        self.assertEqual(c.check_convexity_conditions(), False)
        c = self._object_factory()
        c.r2.lb = -1
        self.assertEqual(c.check_convexity_conditions(), False)
        c = self._object_factory()
        c.x[0].domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(), False)
        self.assertEqual(c.check_convexity_conditions(relax=True), True)

    def test_as_domain(self):
        ret = rotated_quadratic.as_domain(r1=3, r2=4, x=[1, 2])
        self.assertIs(type(ret), block)
        q, c, r1, r2, x = (ret.q, ret.c, ret.r1, ret.r2, ret.x)
        self.assertEqual(q.check_convexity_conditions(), True)
        self.assertIs(type(q), rotated_quadratic)
        self.assertIs(type(x), variable_tuple)
        self.assertIs(type(r1), variable)
        self.assertIs(type(r2), variable)
        self.assertEqual(len(x), 2)
        self.assertIs(type(c), constraint_tuple)
        self.assertEqual(len(c), 4)
        self.assertEqual(c[0].rhs, 3)
        r1.value = 3
        self.assertEqual(c[0].slack, 0)
        r1.value = None
        self.assertEqual(c[1].rhs, 4)
        r2.value = 4
        self.assertEqual(c[1].slack, 0)
        r2.value = None
        self.assertEqual(c[2].rhs, 1)
        x[0].value = 1
        self.assertEqual(c[2].slack, 0)
        x[0].value = None
        self.assertEqual(c[3].rhs, 2)
        x[1].value = 2
        self.assertEqual(c[3].slack, 0)
        x[1].value = None