import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.logical_expr import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error, check_model_algebraic
from pyomo.gdp.plugins.partition_disjuncts import (
from pyomo.core import Block, value
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.common_tests as ct
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers
class LogicalExpressions(unittest.TestCase, CommonTests):

    def test_logical_constraints_on_disjunct_copied(self):
        m = models.makeLogicalConstraintsOnDisjuncts_NonlinearConvex()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x], [m.y]], compute_bounds_method=compute_fbbt_bounds)
        d1 = m.d[1].transformation_block
        self.assertEqual(len(d1.component_map(LogicalConstraint)), 1)
        c = d1.component('logical_constraints')
        self.assertIsInstance(c, LogicalConstraint)
        self.assertEqual(len(c), 1)
        self.assertIsInstance(c[1].expr, NotExpression)
        self.assertIs(c[1].expr.args[0], m.Y[1])
        d2 = m.d[2].transformation_block
        self.assertEqual(len(d2.component_map(LogicalConstraint)), 1)
        c = d2.component('logical_constraints')
        self.assertIsInstance(c, LogicalConstraint)
        self.assertEqual(len(c), 1)
        self.assertIsInstance(c[1].expr, AndExpression)
        self.assertEqual(len(c[1].expr.args), 2)
        self.assertIs(c[1].expr.args[0], m.Y[1])
        self.assertIs(c[1].expr.args[1], m.Y[2])
        d3 = m.d[3].transformation_block
        self.assertEqual(len(d3.component_map(LogicalConstraint)), 1)
        c = d3.component('logical_constraints')
        self.assertEqual(len(c), 0)
        d4 = m.d[4].transformation_block
        self.assertEqual(len(d4.component_map(LogicalConstraint)), 1)
        c = d4.component('logical_constraints')
        self.assertIsInstance(c, LogicalConstraint)
        self.assertEqual(len(c), 2)
        self.assertIsInstance(c[1].expr, ExactlyExpression)
        self.assertEqual(len(c[1].expr.args), 2)
        self.assertEqual(c[1].expr.args[0], 1)
        self.assertIs(c[1].expr.args[1], m.Y[1])
        self.assertIsInstance(c[2].expr, NotExpression)
        self.assertIs(c[2].expr.args[0], m.Y[2])

    @unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
    def test_original_indicator_vars_in_logical_constraints(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        TransformationFactory('gdp.between_steps').apply_to(m, variable_partitions=[[m.x]], compute_bounds_method=compute_fbbt_bounds)
        self.assertTrue(check_model_algebraic(m))
        SolverFactory('gurobi_direct').solve(m)
        self.assertAlmostEqual(value(m.x), 8)
        self.assertFalse(value(m.d[1].indicator_var))
        self.assertTrue(value(m.d[2].indicator_var))
        self.assertTrue(value(m.d[3].indicator_var))
        self.assertFalse(value(m.d[4].indicator_var))