from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
class TestSpecialCases(unittest.TestCase):

    def test_local_vars(self):
        """checks that if nothing is marked as local, we assume it is all
        global. We disaggregate everything to be safe."""
        m = ConcreteModel()
        m.x = Var(bounds=(5, 100))
        m.y = Var(bounds=(0, 100))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.y >= m.x)
        m.d2 = Disjunct()
        m.d2.z = Var()
        m.d2.c = Constraint(expr=m.y >= m.d2.z)
        m.disj = Disjunction(expr=[m.d1, m.d2])
        self.assertRaisesRegex(GDP_Error, '.*Missing bound for d2.z.*', TransformationFactory('gdp.hull').create_using, m)
        m.d2.z.setlb(7)
        self.assertRaisesRegex(GDP_Error, '.*Missing bound for d2.z.*', TransformationFactory('gdp.hull').create_using, m)
        m.d2.z.setub(9)
        i = TransformationFactory('gdp.hull').create_using(m)
        rd = i._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1]
        varBlock = rd.disaggregatedVars
        self.assertEqual(sorted(varBlock.component_map(Var)), ['d2.z', 'y'])
        self.assertEqual(len(rd.component_map(Constraint)), 3)
        self.assertEqual(i.d2.z.bounds, (7, 9))
        z = varBlock.component('d2.z')
        self.assertIsInstance(z, Var)
        self.assertEqual(z.bounds, (0, 9))
        z_bounds = rd.component('d2.z_bounds')
        self.assertEqual(len(z_bounds), 2)
        self.assertEqual(z_bounds['lb'].lower, None)
        self.assertEqual(z_bounds['lb'].upper, 0)
        self.assertEqual(z_bounds['ub'].lower, None)
        self.assertEqual(z_bounds['ub'].upper, 0)
        i.d2.indicator_var = True
        z.set_value(2)
        self.assertEqual(z_bounds['lb'].body(), 5)
        self.assertEqual(z_bounds['ub'].body(), -7)
        m.d2.z.setlb(-9)
        m.d2.z.setub(-7)
        i = TransformationFactory('gdp.hull').create_using(m)
        rd = i._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1]
        varBlock = rd.disaggregatedVars
        self.assertEqual(sorted(varBlock.component_map(Var)), ['d2.z', 'y'])
        self.assertEqual(len(rd.component_map(Constraint)), 3)
        self.assertEqual(i.d2.z.bounds, (-9, -7))
        z = varBlock.component('d2.z')
        self.assertIsInstance(z, Var)
        self.assertEqual(z.bounds, (-9, 0))
        z_bounds = rd.component('d2.z_bounds')
        self.assertEqual(len(z_bounds), 2)
        self.assertEqual(z_bounds['lb'].lower, None)
        self.assertEqual(z_bounds['lb'].upper, 0)
        self.assertEqual(z_bounds['ub'].lower, None)
        self.assertEqual(z_bounds['ub'].upper, 0)
        i.d2.indicator_var = True
        z.set_value(2)
        self.assertEqual(z_bounds['lb'].body(), -11)
        self.assertEqual(z_bounds['ub'].body(), 9)

    def test_local_var_suffix(self):
        hull = TransformationFactory('gdp.hull')
        model = ConcreteModel()
        model.x = Var(bounds=(5, 100))
        model.y = Var(bounds=(0, 100))
        model.d1 = Disjunct()
        model.d1.c = Constraint(expr=model.y >= model.x)
        model.d2 = Disjunct()
        model.d2.z = Var(bounds=(-9, -7))
        model.d2.c = Constraint(expr=model.y >= model.d2.z)
        model.disj = Disjunction(expr=[model.d1, model.d2])
        m = hull.create_using(model)
        self.assertEqual(m.d2.z.lb, -9)
        self.assertEqual(m.d2.z.ub, -7)
        z_disaggregated = m.d2.transformation_block.disaggregatedVars.component('d2.z')
        self.assertIsInstance(z_disaggregated, Var)
        self.assertIs(z_disaggregated, hull.get_disaggregated_var(m.d2.z, m.d2))
        model.d2.LocalVars = Suffix(direction=Suffix.LOCAL)
        model.d2.LocalVars[model.d2] = [model.d2.z]
        m = hull.create_using(model)
        self.assertEqual(m.d2.z.lb, -9)
        self.assertEqual(m.d2.z.ub, 0)
        self.assertIs(hull.get_disaggregated_var(m.d2.z, m.d2), m.d2.z)
        self.assertIsNone(m.d2.transformation_block.disaggregatedVars.component('z'))