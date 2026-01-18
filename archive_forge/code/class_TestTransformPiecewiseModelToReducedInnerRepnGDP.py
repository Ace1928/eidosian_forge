import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var
class TestTransformPiecewiseModelToReducedInnerRepnGDP(unittest.TestCase):

    def check_disjunct(self, d, not_pts):
        self.assertEqual(len(d.component_map(Constraint)), 1)
        self.assertEqual(len(d.component_map(Var)), 1)
        self.assertIsInstance(d.lambdas_zero_for_other_simplices, Constraint)
        self.assertEqual(len(d.lambdas_zero_for_other_simplices), len(not_pts))
        transBlock = d.parent_block()
        for i, cons in zip(not_pts, d.lambdas_zero_for_other_simplices.values()):
            assertExpressionsEqual(self, cons.expr, transBlock.lambdas[i] <= 0)

    def check_log_trans_block_structure(self, transBlock):
        m = transBlock.model()
        self.assertEqual(len(transBlock.component_map(Disjunct)), 1)
        self.assertEqual(len(transBlock.component_map(Disjunction)), 1)
        self.assertEqual(len(transBlock.component_map(Var)), 2)
        self.assertIsInstance(transBlock.substitute_var, Var)
        self.assertIsInstance(transBlock.lambdas, Var)
        self.assertEqual(len(transBlock.lambdas), 4)
        for lamb in transBlock.lambdas.values():
            self.assertEqual(lamb.lb, 0)
            self.assertEqual(lamb.ub, 1)
        self.assertIsInstance(transBlock.convex_combo, Constraint)
        assertExpressionsEqual(self, transBlock.convex_combo.expr, transBlock.lambdas[0] + transBlock.lambdas[1] + transBlock.lambdas[2] + transBlock.lambdas[3] == 1)
        self.assertIsInstance(transBlock.linear_combo, Constraint)
        self.assertEqual(len(transBlock.linear_combo), 1)
        pts = m.pw_log._points
        assertExpressionsEqual(self, transBlock.linear_combo[0].expr, m.x == pts[0][0] * transBlock.lambdas[0] + pts[1][0] * transBlock.lambdas[1] + pts[2][0] * transBlock.lambdas[2] + pts[3][0] * transBlock.lambdas[3])
        self.assertIsInstance(transBlock.linear_func, Constraint)
        self.assertEqual(len(transBlock.linear_func), 1)
        assertExpressionsEqual(self, transBlock.linear_func.expr, transBlock.lambdas[0] * m.f1(1) + transBlock.lambdas[1] * m.f1(3) + transBlock.lambdas[2] * m.f2(6) + transBlock.lambdas[3] * m.f3(10) == transBlock.substitute_var, places=7)

    def check_paraboloid_trans_block_structure(self, transBlock):
        m = transBlock.model()
        self.assertEqual(len(transBlock.component_map(Disjunct)), 1)
        self.assertEqual(len(transBlock.component_map(Disjunction)), 1)
        self.assertEqual(len(transBlock.component_map(Var)), 2)
        self.assertEqual(len(transBlock.component_map(Constraint)), 3)
        self.assertIsInstance(transBlock.substitute_var, Var)
        self.assertIsInstance(transBlock.lambdas, Var)
        self.assertEqual(len(transBlock.lambdas), 6)
        for lamb in transBlock.lambdas.values():
            self.assertEqual(lamb.lb, 0)
            self.assertEqual(lamb.ub, 1)
        self.assertIsInstance(transBlock.convex_combo, Constraint)
        assertExpressionsEqual(self, transBlock.convex_combo.expr, transBlock.lambdas[0] + transBlock.lambdas[1] + transBlock.lambdas[2] + transBlock.lambdas[3] + transBlock.lambdas[4] + transBlock.lambdas[5] == 1)
        self.assertIsInstance(transBlock.linear_combo, Constraint)
        self.assertEqual(len(transBlock.linear_combo), 2)
        pts = m.pw_paraboloid._points
        assertExpressionsEqual(self, transBlock.linear_combo[0].expr, m.x1 == pts[0][0] * transBlock.lambdas[0] + pts[1][0] * transBlock.lambdas[1] + pts[2][0] * transBlock.lambdas[2] + pts[3][0] * transBlock.lambdas[3] + pts[4][0] * transBlock.lambdas[4] + pts[5][0] * transBlock.lambdas[5])
        assertExpressionsEqual(self, transBlock.linear_combo[1].expr, m.x2 == pts[0][1] * transBlock.lambdas[0] + pts[1][1] * transBlock.lambdas[1] + pts[2][1] * transBlock.lambdas[2] + pts[3][1] * transBlock.lambdas[3] + pts[4][1] * transBlock.lambdas[4] + pts[5][1] * transBlock.lambdas[5])
        self.assertIsInstance(transBlock.linear_func, Constraint)
        self.assertEqual(len(transBlock.linear_func), 1)
        assertExpressionsEqual(self, transBlock.linear_func.expr, transBlock.lambdas[0] * m.g1(0, 1) + transBlock.lambdas[1] * m.g1(0, 4) + transBlock.lambdas[2] * m.g1(3, 4) + transBlock.lambdas[3] * m.g1(3, 1) + transBlock.lambdas[4] * m.g2(3, 7) + transBlock.lambdas[5] * m.g2(0, 7) == transBlock.substitute_var)

    def check_pw_log(self, m):
        z = m.pw_log.get_transformation_var(m.log_expr)
        self.assertIsInstance(z, Var)
        log_block = z.parent_block()
        self.check_log_trans_block_structure(log_block)
        self.assertEqual(len(log_block.disjuncts), 3)
        disjuncts_dict = {log_block.disjuncts[0]: (2, 3), log_block.disjuncts[1]: (0, 3), log_block.disjuncts[2]: (0, 1)}
        for d, not_pts in disjuncts_dict.items():
            self.check_disjunct(d, not_pts)
        self.assertIsInstance(log_block.pick_a_piece, Disjunction)
        self.assertEqual(len(log_block.pick_a_piece.disjuncts), 3)
        for i in range(2):
            self.assertIs(log_block.pick_a_piece.disjuncts[i], log_block.disjuncts[i])
        self.assertIs(m.obj.expr.expr, log_block.substitute_var)

    def check_pw_paraboloid(self, m):
        z = m.pw_paraboloid.get_transformation_var(m.paraboloid_expr)
        self.assertIsInstance(z, Var)
        paraboloid_block = z.parent_block()
        self.check_paraboloid_trans_block_structure(paraboloid_block)
        self.assertEqual(len(paraboloid_block.disjuncts), 4)
        disjuncts_dict = {paraboloid_block.disjuncts[0]: [3, 4, 5], paraboloid_block.disjuncts[1]: [1, 4, 5], paraboloid_block.disjuncts[2]: [0, 1, 3], paraboloid_block.disjuncts[3]: [0, 3, 4]}
        for d, not_pts in disjuncts_dict.items():
            self.check_disjunct(d, not_pts)
        self.assertIsInstance(paraboloid_block.pick_a_piece, Disjunction)
        self.assertEqual(len(paraboloid_block.pick_a_piece.disjuncts), 4)
        for i in range(3):
            self.assertIs(paraboloid_block.pick_a_piece.disjuncts[i], paraboloid_block.disjuncts[i])
        self.assertIs(m.indexed_c[0].body.args[0].expr, paraboloid_block.substitute_var)

    def test_transformation_do_not_descend(self):
        ct.check_transformation_do_not_descend(self, 'contrib.piecewise.reduced_inner_repn_gdp')

    def test_transformation_PiecewiseLinearFunction_targets(self):
        ct.check_transformation_PiecewiseLinearFunction_targets(self, 'contrib.piecewise.reduced_inner_repn_gdp')

    def test_descend_into_expressions(self):
        ct.check_descend_into_expressions(self, 'contrib.piecewise.reduced_inner_repn_gdp')

    def test_descend_into_expressions_constraint_target(self):
        ct.check_descend_into_expressions_constraint_target(self, 'contrib.piecewise.reduced_inner_repn_gdp')

    def test_descend_into_expressions_objective_target(self):
        ct.check_descend_into_expressions_objective_target(self, 'contrib.piecewise.reduced_inner_repn_gdp')

    @unittest.skipUnless(SolverFactory('gurobi').available(), 'Gurobi is not available')
    @unittest.skipUnless(SolverFactory('gurobi').license_is_valid(), 'No license')
    def test_solve_convex_combo_model(self):
        m = models.make_log_x_model()
        TransformationFactory('contrib.piecewise.convex_combination').apply_to(m)
        SolverFactory('gurobi').solve(m)
        ct.check_log_x_model_soln(self, m)