from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
class EstimatingMwithFixedVars(unittest.TestCase):

    def test_tighter_Ms_when_vars_fixed_forever(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(bounds=(0, 70))
        m.d = Disjunct()
        m.d.c = Constraint(expr=m.x + m.y <= 13)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x >= 7)
        m.disj = Disjunction(expr=[m.d, m.d2])
        m.y.fix(10)
        bigm = TransformationFactory('gdp.bigm')
        promise = bigm.create_using(m, assume_fixed_vars_permanent=True)
        bigm.apply_to(m, assume_fixed_vars_permanent=False)
        xformed = bigm.get_transformed_constraints(m.d.c)
        self.assertEqual(len(xformed), 1)
        cons = xformed[0]
        self.assertEqual(cons.upper, 13)
        self.assertIsNone(cons.lower)
        repn = generate_standard_repn(cons.body)
        self.assertEqual(repn.constant, -57)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, m.x, 1)
        ct.check_linear_coef(self, repn, m.d.indicator_var, 67)
        xformed = bigm.get_transformed_constraints(promise.d.c)
        self.assertEqual(len(xformed), 1)
        cons = xformed[0]
        self.assertEqual(cons.upper, 13)
        self.assertIsNone(cons.lower)
        repn = generate_standard_repn(cons.body)
        self.assertEqual(repn.constant, 3)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, promise.x, 1)
        ct.check_linear_coef(self, repn, promise.d.indicator_var, 7)