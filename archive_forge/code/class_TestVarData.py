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
class TestVarData(unittest.TestCase):

    def test_lower_bound(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(mutable=True, initialize=2)
        self.assertIsNone(m.x.lower)
        m.x.domain = NonNegativeReals
        self.assertIs(type(m.x.lower), int)
        self.assertEqual(value(m.x.lower), 0)
        m.x.domain = Reals
        m.x.setlb(5 * m.p)
        self.assertIs(type(m.x.lower), NPV_ProductExpression)
        self.assertEqual(value(m.x.lower), 10)
        m.x.domain = NonNegativeReals
        self.assertIs(type(m.x.lower), NPV_MaxExpression)
        self.assertEqual(value(m.x.lower), 10)
        with self.assertRaisesRegex(ValueError, "Potentially variable input of type 'ScalarVar' supplied as lower bound for variable 'x'"):
            m.x.setlb(m.x)

    def test_lower_bound_setter(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertIsNone(m.x.lb)
        m.x.lb = 1
        self.assertEqual(m.x.lb, 1)
        m.x.lower = 2
        self.assertEqual(m.x.lb, 2)
        m.x.setlb(3)
        self.assertEqual(m.x.lb, 3)
        m.y = Var([1])
        self.assertIsNone(m.y[1].lb)
        m.y[1].lb = 1
        self.assertEqual(m.y[1].lb, 1)
        m.y[1].lower = 2
        self.assertEqual(m.y[1].lb, 2)
        m.y[1].setlb(3)
        self.assertEqual(m.y[1].lb, 3)

    def test_upper_bound(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(mutable=True, initialize=2)
        self.assertIsNone(m.x.upper)
        m.x.domain = NonPositiveReals
        self.assertIs(type(m.x.upper), int)
        self.assertEqual(value(m.x.upper), 0)
        m.x.domain = Reals
        m.x.setub(-5 * m.p)
        self.assertIs(type(m.x.upper), NPV_ProductExpression)
        self.assertEqual(value(m.x.upper), -10)
        m.x.domain = NonPositiveReals
        self.assertIs(type(m.x.upper), NPV_MinExpression)
        self.assertEqual(value(m.x.upper), -10)
        with self.assertRaisesRegex(ValueError, "Potentially variable input of type 'ScalarVar' supplied as upper bound for variable 'x'"):
            m.x.setub(m.x)

    def test_upper_bound_setter(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertIsNone(m.x.ub)
        m.x.ub = 1
        self.assertEqual(m.x.ub, 1)
        m.x.upper = 2
        self.assertEqual(m.x.ub, 2)
        m.x.setub(3)
        self.assertEqual(m.x.ub, 3)
        m.y = Var([1])
        self.assertIsNone(m.y[1].ub)
        m.y[1].ub = 1
        self.assertEqual(m.y[1].ub, 1)
        m.y[1].upper = 2
        self.assertEqual(m.y[1].ub, 2)
        m.y[1].setub(3)
        self.assertEqual(m.y[1].ub, 3)

    def test_lb(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertEqual(m.x.lb, None)
        m.x.domain = NonNegativeReals
        self.assertEqual(m.x.lb, 0)
        m.x.lb = float('inf')
        with self.assertRaisesRegex(ValueError, 'invalid non-finite lower bound \\(inf\\)'):
            m.x.lb
        m.x.lb = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid non-finite lower bound \\(nan\\)'):
            m.x.lb

    def test_ub(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertEqual(m.x.ub, None)
        m.x.domain = NonPositiveReals
        self.assertEqual(m.x.ub, 0)
        m.x.ub = float('-inf')
        with self.assertRaisesRegex(ValueError, 'invalid non-finite upper bound \\(-inf\\)'):
            m.x.ub
        m.x.ub = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid non-finite upper bound \\(nan\\)'):
            m.x.ub

    def test_bounds(self):
        m = ConcreteModel()
        m.x = Var()
        lb, ub = m.x.bounds
        self.assertEqual(lb, None)
        self.assertEqual(ub, None)
        m.x.domain = NonNegativeReals
        lb, ub = m.x.bounds
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)
        m.x.lb = float('inf')
        with self.assertRaisesRegex(ValueError, 'invalid non-finite lower bound \\(inf\\)'):
            lb, ub = m.x.bounds
        m.x.lb = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid non-finite lower bound \\(nan\\)'):
            lb, ub = m.x.bounds
        m.x.lb = None
        m.x.domain = NonPositiveReals
        lb, ub = m.x.bounds
        self.assertEqual(lb, None)
        self.assertEqual(ub, 0)
        m.x.ub = float('-inf')
        with self.assertRaisesRegex(ValueError, 'invalid non-finite upper bound \\(-inf\\)'):
            lb, ub = m.x.bounds
        m.x.ub = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid non-finite upper bound \\(nan\\)'):
            lb, ub = m.x.bounds