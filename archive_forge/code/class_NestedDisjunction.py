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
class NestedDisjunction(unittest.TestCase, CommonTests):

    def setUp(self):
        random.seed(666)

    def test_disjuncts_inactive(self):
        ct.check_disjuncts_inactive_nested(self, 'hull')

    def test_deactivated_disjunct_leaves_nested_disjuncts_active(self):
        ct.check_deactivated_disjunct_leaves_nested_disjunct_active(self, 'hull')

    def test_mappings_between_disjunctions_and_xors(self):
        m = models.makeNestedDisjunctions()
        transform = TransformationFactory('gdp.hull')
        transform.apply_to(m)
        transBlock = m.component('_pyomo_gdp_hull_reformulation')
        disjunctionPairs = [(m.disjunction, transBlock.disjunction_xor), (m.disjunct[1].innerdisjunction[0], m.disjunct[1].innerdisjunction[0].algebraic_constraint.parent_block().innerdisjunction_xor[0]), (m.simpledisjunct.innerdisjunction, m.simpledisjunct.innerdisjunction.algebraic_constraint.parent_block().innerdisjunction_xor)]
        for disjunction, xor in disjunctionPairs:
            self.assertIs(disjunction.algebraic_constraint, xor)
            self.assertIs(transform.get_src_disjunction(xor), disjunction)

    def test_unique_reference_to_nested_indicator_var(self):
        ct.check_unique_reference_to_nested_indicator_var(self, 'hull')

    def test_disjunct_targets_inactive(self):
        ct.check_disjunct_targets_inactive(self, 'hull')

    def test_disjunct_only_targets_transformed(self):
        ct.check_disjunct_only_targets_transformed(self, 'hull')

    def test_disjunctData_targets_inactive(self):
        ct.check_disjunctData_targets_inactive(self, 'hull')

    def test_disjunctData_only_targets_transformed(self):
        ct.check_disjunctData_only_targets_transformed(self, 'hull')

    def test_disjunction_target_err(self):
        ct.check_disjunction_target_err(self, 'hull')

    def test_nested_disjunction_target(self):
        ct.check_nested_disjunction_target(self, 'hull')

    def test_target_appears_twice(self):
        ct.check_target_appears_twice(self, 'hull')

    @unittest.skipIf(not linear_solvers, 'No linear solver available')
    def test_relaxation_feasibility(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        TransformationFactory('gdp.hull').apply_to(m)
        solver = SolverFactory(linear_solvers[0])
        cases = [(True, True, True, True, None), (False, False, False, False, None), (True, False, False, False, None), (False, True, False, False, 1.1), (False, False, True, False, None), (False, False, False, True, None), (True, True, False, False, None), (True, False, True, False, 1.2), (True, False, False, True, 1.3), (True, False, True, True, None)]
        for case in cases:
            m.d1.indicator_var.fix(case[0])
            m.d2.indicator_var.fix(case[1])
            m.d3.indicator_var.fix(case[2])
            m.d4.indicator_var.fix(case[3])
            results = solver.solve(m)
            if case[4] is None:
                self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)
            else:
                self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
                self.assertEqual(value(m.obj), case[4])

    @unittest.skipIf(not linear_solvers, 'No linear solver available')
    def test_relaxation_feasibility_transform_inner_first(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        TransformationFactory('gdp.hull').apply_to(m.d1)
        TransformationFactory('gdp.hull').apply_to(m)
        solver = SolverFactory(linear_solvers[0])
        cases = [(True, True, True, True, None), (False, False, False, False, None), (True, False, False, False, None), (False, True, False, False, 1.1), (False, False, True, False, None), (False, False, False, True, None), (True, True, False, False, None), (True, False, True, False, 1.2), (True, False, False, True, 1.3), (True, False, True, True, None)]
        for case in cases:
            m.d1.indicator_var.fix(case[0])
            m.d2.indicator_var.fix(case[1])
            m.d3.indicator_var.fix(case[2])
            m.d4.indicator_var.fix(case[3])
            results = solver.solve(m)
            if case[4] is None:
                self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)
            else:
                self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
                self.assertEqual(value(m.obj), case[4])

    def test_create_using(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        self.diff_apply_to_and_create_using(m)

    def check_outer_disaggregation_constraint(self, cons, var, disj1, disj2, rhs=None):
        if rhs is None:
            rhs = var
        hull = TransformationFactory('gdp.hull')
        self.assertTrue(cons.active)
        self.assertEqual(cons.lower, 0)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        ct.check_linear_coef(self, repn, rhs, 1)
        ct.check_linear_coef(self, repn, hull.get_disaggregated_var(var, disj1), -1)
        ct.check_linear_coef(self, repn, hull.get_disaggregated_var(var, disj2), -1)

    def check_bounds_constraint_ub(self, constraint, ub, dis_var, ind_var):
        hull = TransformationFactory('gdp.hull')
        self.assertIsInstance(constraint, Constraint)
        self.assertTrue(constraint.active)
        self.assertEqual(len(constraint), 1)
        self.assertTrue(constraint['ub'].active)
        self.assertEqual(constraint['ub'].upper, 0)
        self.assertIsNone(constraint['ub'].lower)
        repn = generate_standard_repn(constraint['ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, dis_var, 1)
        ct.check_linear_coef(self, repn, ind_var, -ub)
        self.assertIs(constraint, hull.get_var_bounds_constraint(dis_var))

    def check_transformed_constraint(self, cons, dis, lb, ind_var):
        hull = TransformationFactory('gdp.hull')
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertTrue(cons.active)
        self.assertIsNone(cons.lower)
        self.assertEqual(value(cons.upper), 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, dis, -1)
        ct.check_linear_coef(self, repn, ind_var, lb)
        orig = ind_var.parent_block().c
        self.assertIs(hull.get_src_constraint(cons), orig)

    def test_transformed_model_nestedDisjuncts(self):
        m = models.makeNestedDisjunctions_NestedDisjuncts()
        m.LocalVars = Suffix(direction=Suffix.LOCAL)
        m.LocalVars[m.d1] = [m.d1.binary_indicator_var, m.d1.d3.binary_indicator_var, m.d1.d4.binary_indicator_var]
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        self.check_transformed_model_nestedDisjuncts(m, m.d1.d3.binary_indicator_var, m.d1.d4.binary_indicator_var)
        all_cons = list(m.component_data_objects(Constraint, active=True, descend_into=Block))
        self.assertEqual(len(all_cons), 16)

    def check_transformed_model_nestedDisjuncts(self, m, d3, d4):
        hull = TransformationFactory('gdp.hull')
        transBlock = m._pyomo_gdp_hull_reformulation
        self.assertTrue(transBlock.active)
        xor = transBlock.disj_xor
        self.assertIsInstance(xor, Constraint)
        ct.check_obj_in_active_tree(self, xor)
        assertExpressionsEqual(self, xor.expr, m.d1.binary_indicator_var + m.d2.binary_indicator_var == 1)
        self.assertIs(xor, m.disj.algebraic_constraint)
        self.assertIs(m.disj, hull.get_src_disjunction(xor))
        xor = m.d1.disj2.algebraic_constraint
        self.assertIs(m.d1.disj2, hull.get_src_disjunction(xor))
        xor = hull.get_transformed_constraints(xor)
        self.assertEqual(len(xor), 1)
        xor = xor[0]
        ct.check_obj_in_active_tree(self, xor)
        xor_expr = self.simplify_cons(xor)
        assertExpressionsEqual(self, xor_expr, d3 + d4 - m.d1.binary_indicator_var == 0.0)
        x_d3 = hull.get_disaggregated_var(m.x, m.d1.d3)
        x_d4 = hull.get_disaggregated_var(m.x, m.d1.d4)
        x_d1 = hull.get_disaggregated_var(m.x, m.d1)
        x_d2 = hull.get_disaggregated_var(m.x, m.d2)
        for x in [x_d1, x_d2, x_d3, x_d4]:
            self.assertEqual(x.lb, 0)
            self.assertEqual(x.ub, 2)
        cons = hull.get_disaggregation_constraint(m.x, m.d1.disj2)
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, x_d1 - x_d3 - x_d4 == 0.0)
        cons = hull.get_disaggregation_constraint(m.x, m.disj)
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, m.x - x_d1 - x_d2 == 0.0)
        cons = hull.get_transformed_constraints(m.d1.d3.c)
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_leq_cons(cons)
        assertExpressionsEqual(self, cons_expr, 1.2 * d3 - x_d3 <= 0.0)
        cons = hull.get_transformed_constraints(m.d1.d4.c)
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_leq_cons(cons)
        assertExpressionsEqual(self, cons_expr, 1.3 * d4 - x_d4 <= 0.0)
        cons = hull.get_transformed_constraints(m.d1.c)
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_leq_cons(cons)
        assertExpressionsEqual(self, cons_expr, 1.0 * m.d1.binary_indicator_var - x_d1 <= 0.0)
        cons = hull.get_transformed_constraints(m.d2.c)
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_leq_cons(cons)
        assertExpressionsEqual(self, cons_expr, 1.1 * m.d2.binary_indicator_var - x_d2 <= 0.0)
        cons = hull.get_var_bounds_constraint(x_d1)
        self.assertEqual(len(cons), 1)
        ct.check_obj_in_active_tree(self, cons['ub'])
        cons_expr = self.simplify_leq_cons(cons['ub'])
        assertExpressionsEqual(self, cons_expr, x_d1 - 2 * m.d1.binary_indicator_var <= 0.0)
        cons = hull.get_var_bounds_constraint(x_d2)
        self.assertEqual(len(cons), 1)
        ct.check_obj_in_active_tree(self, cons['ub'])
        cons_expr = self.simplify_leq_cons(cons['ub'])
        assertExpressionsEqual(self, cons_expr, x_d2 - 2 * m.d2.binary_indicator_var <= 0.0)
        cons = hull.get_var_bounds_constraint(x_d3, m.d1.d3)
        self.assertEqual(len(cons), 1)
        cons = hull.get_transformed_constraints(cons['ub'])
        self.assertEqual(len(cons), 1)
        ub = cons[0]
        ct.check_obj_in_active_tree(self, ub)
        cons_expr = self.simplify_leq_cons(ub)
        assertExpressionsEqual(self, cons_expr, x_d3 - 2 * d3 <= 0.0)
        cons = hull.get_var_bounds_constraint(x_d4, m.d1.d4)
        self.assertEqual(len(cons), 1)
        cons = hull.get_transformed_constraints(cons['ub'])
        self.assertEqual(len(cons), 1)
        ub = cons[0]
        ct.check_obj_in_active_tree(self, ub)
        cons_expr = self.simplify_leq_cons(ub)
        assertExpressionsEqual(self, cons_expr, x_d4 - 2 * d4 <= 0.0)
        cons = hull.get_var_bounds_constraint(x_d3, m.d1)
        self.assertEqual(len(cons), 1)
        ub = cons['ub']
        ct.check_obj_in_active_tree(self, ub)
        cons_expr = self.simplify_leq_cons(ub)
        assertExpressionsEqual(self, cons_expr, x_d3 - 2 * m.d1.binary_indicator_var <= 0.0)
        cons = hull.get_var_bounds_constraint(x_d4, m.d1)
        self.assertEqual(len(cons), 1)
        ub = cons['ub']
        ct.check_obj_in_active_tree(self, ub)
        cons_expr = self.simplify_leq_cons(ub)
        assertExpressionsEqual(self, cons_expr, x_d4 - 2 * m.d1.binary_indicator_var <= 0.0)
        cons = hull.get_var_bounds_constraint(d3)
        ct.check_obj_in_active_tree(self, cons['ub'])
        assertExpressionsEqual(self, cons['ub'].expr, d3 <= m.d1.binary_indicator_var)
        cons = hull.get_var_bounds_constraint(d4)
        ct.check_obj_in_active_tree(self, cons['ub'])
        assertExpressionsEqual(self, cons['ub'].expr, d4 <= m.d1.binary_indicator_var)

    @unittest.skipIf(not linear_solvers, 'No linear solver available')
    def test_solve_nested_model(self):
        m = models.makeNestedDisjunctions_NestedDisjuncts()
        m.LocalVars = Suffix(direction=Suffix.LOCAL)
        m.LocalVars[m.d1] = [m.d1.binary_indicator_var, m.d1.d3.binary_indicator_var, m.d1.d4.binary_indicator_var]
        hull = TransformationFactory('gdp.hull')
        m_hull = hull.create_using(m)
        SolverFactory(linear_solvers[0]).solve(m_hull)
        self.assertEqual(value(m_hull.d1.binary_indicator_var), 0)
        self.assertEqual(value(m_hull.d2.binary_indicator_var), 1)
        self.assertEqual(value(m_hull.x), 1.1)
        TransformationFactory('gdp.bigm').apply_to(m, targets=m.d1.disj2)
        hull.apply_to(m)
        SolverFactory(linear_solvers[0]).solve(m)
        self.assertEqual(value(m.d1.binary_indicator_var), 0)
        self.assertEqual(value(m.d2.binary_indicator_var), 1)
        self.assertEqual(value(m.x), 1.1)

    @unittest.skipIf(not linear_solvers, 'No linear solver available')
    def test_disaggregated_vars_are_set_to_0_correctly(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        m.d1.indicator_var.fix(False)
        m.d2.indicator_var.fix(True)
        m.d3.indicator_var.fix(False)
        m.d4.indicator_var.fix(False)
        results = SolverFactory(linear_solvers[0]).solve(m)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
        self.assertEqual(value(m.x), 1.1)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d1)), 0)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d2)), 1.1)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d3)), 0)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d4)), 0)
        m.d1.indicator_var.fix(True)
        m.d2.indicator_var.fix(False)
        m.d3.indicator_var.fix(True)
        m.d4.indicator_var.fix(False)
        results = SolverFactory(linear_solvers[0]).solve(m)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
        self.assertEqual(value(m.x), 1.2)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d1)), 1.2)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d2)), 0)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d3)), 1.2)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d4)), 0)

    def test_nested_with_local_vars(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.S = RangeSet(2)

        @m.Disjunct()
        def d_l(d):
            d.lambdas = Var(m.S, bounds=(0, 1))
            d.LocalVars = Suffix(direction=Suffix.LOCAL)
            d.LocalVars[d] = list(d.lambdas.values())
            d.c1 = Constraint(expr=d.lambdas[1] + d.lambdas[2] == 1)
            d.c2 = Constraint(expr=m.x == 2 * d.lambdas[1] + 3 * d.lambdas[2])

        @m.Disjunct()
        def d_r(d):

            @d.Disjunct()
            def d_l(e):
                e.lambdas = Var(m.S, bounds=(0, 1))
                e.LocalVars = Suffix(direction=Suffix.LOCAL)
                e.LocalVars[e] = list(e.lambdas.values())
                e.c1 = Constraint(expr=e.lambdas[1] + e.lambdas[2] == 1)
                e.c2 = Constraint(expr=m.x == 2 * e.lambdas[1] + 3 * e.lambdas[2])

            @d.Disjunct()
            def d_r(e):
                e.lambdas = Var(m.S, bounds=(0, 1))
                e.LocalVars = Suffix(direction=Suffix.LOCAL)
                e.LocalVars[e] = list(e.lambdas.values())
                e.c1 = Constraint(expr=e.lambdas[1] + e.lambdas[2] == 1)
                e.c2 = Constraint(expr=m.x == 2 * e.lambdas[1] + 3 * e.lambdas[2])
            d.LocalVars = Suffix(direction=Suffix.LOCAL)
            d.LocalVars[d] = [d.d_l.indicator_var.get_associated_binary(), d.d_r.indicator_var.get_associated_binary()]
            d.inner_disj = Disjunction(expr=[d.d_l, d.d_r])
        m.disj = Disjunction(expr=[m.d_l, m.d_r])
        m.obj = Objective(expr=m.x)
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        x1 = hull.get_disaggregated_var(m.x, m.d_l)
        x2 = hull.get_disaggregated_var(m.x, m.d_r)
        x3 = hull.get_disaggregated_var(m.x, m.d_r.d_l)
        x4 = hull.get_disaggregated_var(m.x, m.d_r.d_r)
        for d, x in [(m.d_l, x1), (m.d_r.d_l, x3), (m.d_r.d_r, x4)]:
            lambda1 = hull.get_disaggregated_var(d.lambdas[1], d)
            self.assertIs(lambda1, d.lambdas[1])
            lambda2 = hull.get_disaggregated_var(d.lambdas[2], d)
            self.assertIs(lambda2, d.lambdas[2])
            cons = hull.get_transformed_constraints(d.c1)
            self.assertEqual(len(cons), 1)
            convex_combo = cons[0]
            convex_combo_expr = self.simplify_cons(convex_combo)
            assertExpressionsEqual(self, convex_combo_expr, lambda1 + lambda2 - d.indicator_var.get_associated_binary() == 0.0)
            cons = hull.get_transformed_constraints(d.c2)
            self.assertEqual(len(cons), 1)
            get_x = cons[0]
            get_x_expr = self.simplify_cons(get_x)
            assertExpressionsEqual(self, get_x_expr, x - 2 * lambda1 - 3 * lambda2 == 0.0)
        cons = hull.get_disaggregation_constraint(m.x, m.disj)
        assertExpressionsEqual(self, cons.expr, m.x == x1 + x2)
        cons = hull.get_disaggregation_constraint(m.x, m.d_r.inner_disj)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, x2 - x3 - x4 == 0.0)

    def test_nested_with_var_that_does_not_appear_in_every_disjunct(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(bounds=(-4, 5))
        m.parent1 = Disjunct()
        m.parent2 = Disjunct()
        m.parent2.c = Constraint(expr=m.x == 0)
        m.parent_disjunction = Disjunction(expr=[m.parent1, m.parent2])
        m.child1 = Disjunct()
        m.child1.c = Constraint(expr=m.x <= 8)
        m.child2 = Disjunct()
        m.child2.c = Constraint(expr=m.x + m.y <= 3)
        m.child3 = Disjunct()
        m.child3.c = Constraint(expr=m.x <= 7)
        m.parent1.disjunction = Disjunction(expr=[m.child1, m.child2, m.child3])
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        y_c2 = hull.get_disaggregated_var(m.y, m.child2)
        self.assertEqual(y_c2.bounds, (-4, 5))
        other_y = hull.get_disaggregated_var(m.y, m.child1)
        self.assertEqual(other_y.bounds, (-4, 5))
        other_other_y = hull.get_disaggregated_var(m.y, m.child3)
        self.assertIs(other_y, other_other_y)
        y_p1 = hull.get_disaggregated_var(m.y, m.parent1)
        self.assertEqual(y_p1.bounds, (-4, 5))
        y_p2 = hull.get_disaggregated_var(m.y, m.parent2)
        self.assertEqual(y_p2.bounds, (-4, 5))
        y_cons = hull.get_disaggregation_constraint(m.y, m.parent1.disjunction)
        y_cons_expr = self.simplify_cons(y_cons)
        assertExpressionsEqual(self, y_cons_expr, y_p1 - other_y - y_c2 == 0.0)
        y_cons = hull.get_disaggregation_constraint(m.y, m.parent_disjunction)
        y_cons_expr = self.simplify_cons(y_cons)
        assertExpressionsEqual(self, y_cons_expr, m.y - y_p2 - y_p1 == 0.0)
        x_c1 = hull.get_disaggregated_var(m.x, m.child1)
        x_c2 = hull.get_disaggregated_var(m.x, m.child2)
        x_c3 = hull.get_disaggregated_var(m.x, m.child3)
        x_p1 = hull.get_disaggregated_var(m.x, m.parent1)
        x_p2 = hull.get_disaggregated_var(m.x, m.parent2)
        x_cons_parent = hull.get_disaggregation_constraint(m.x, m.parent_disjunction)
        assertExpressionsEqual(self, x_cons_parent.expr, m.x == x_p1 + x_p2)
        x_cons_child = hull.get_disaggregation_constraint(m.x, m.parent1.disjunction)
        x_cons_child_expr = self.simplify_cons(x_cons_child)
        assertExpressionsEqual(self, x_cons_child_expr, x_p1 - x_c1 - x_c2 - x_c3 == 0.0)

    def simplify_cons(self, cons):
        visitor = LinearRepnVisitor({}, {}, {}, None)
        lb = cons.lower
        ub = cons.upper
        self.assertEqual(cons.lb, cons.ub)
        repn = visitor.walk_expression(cons.body)
        self.assertIsNone(repn.nonlinear)
        return repn.to_expression(visitor) == lb

    def simplify_leq_cons(self, cons):
        visitor = LinearRepnVisitor({}, {}, {}, None)
        self.assertIsNone(cons.lower)
        ub = cons.upper
        repn = visitor.walk_expression(cons.body)
        self.assertIsNone(repn.nonlinear)
        return repn.to_expression(visitor) <= ub

    def test_nested_with_var_that_skips_a_level(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-2, 9))
        m.y = Var(bounds=(-3, 8))
        m.y1 = Disjunct()
        m.y1.c1 = Constraint(expr=m.x >= 4)
        m.y1.z1 = Disjunct()
        m.y1.z1.c1 = Constraint(expr=m.y == 2)
        m.y1.z1.w1 = Disjunct()
        m.y1.z1.w1.c1 = Constraint(expr=m.x == 3)
        m.y1.z1.w2 = Disjunct()
        m.y1.z1.w2.c1 = Constraint(expr=m.x >= 1)
        m.y1.z1.disjunction = Disjunction(expr=[m.y1.z1.w1, m.y1.z1.w2])
        m.y1.z2 = Disjunct()
        m.y1.z2.c1 = Constraint(expr=m.y == 1)
        m.y1.disjunction = Disjunction(expr=[m.y1.z1, m.y1.z2])
        m.y2 = Disjunct()
        m.y2.c1 = Constraint(expr=m.x == 4)
        m.disjunction = Disjunction(expr=[m.y1, m.y2])
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        x_y1 = hull.get_disaggregated_var(m.x, m.y1)
        x_y2 = hull.get_disaggregated_var(m.x, m.y2)
        x_z1 = hull.get_disaggregated_var(m.x, m.y1.z1)
        x_z2 = hull.get_disaggregated_var(m.x, m.y1.z2)
        x_w1 = hull.get_disaggregated_var(m.x, m.y1.z1.w1)
        x_w2 = hull.get_disaggregated_var(m.x, m.y1.z1.w2)
        y_z1 = hull.get_disaggregated_var(m.y, m.y1.z1)
        y_z2 = hull.get_disaggregated_var(m.y, m.y1.z2)
        y_y1 = hull.get_disaggregated_var(m.y, m.y1)
        y_y2 = hull.get_disaggregated_var(m.y, m.y2)
        cons = hull.get_disaggregation_constraint(m.x, m.y1.z1.disjunction)
        self.assertTrue(cons.active)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, x_z1 - x_w1 - x_w2 == 0.0)
        cons = hull.get_disaggregation_constraint(m.x, m.y1.disjunction)
        self.assertTrue(cons.active)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, x_y1 - x_z2 - x_z1 == 0.0)
        cons = hull.get_disaggregation_constraint(m.x, m.disjunction)
        self.assertTrue(cons.active)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, m.x - x_y1 - x_y2 == 0.0)
        cons = hull.get_disaggregation_constraint(m.y, m.y1.z1.disjunction, raise_exception=False)
        self.assertIsNone(cons)
        cons = hull.get_disaggregation_constraint(m.y, m.y1.disjunction)
        self.assertTrue(cons.active)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, y_y1 - y_z1 - y_z2 == 0.0)
        cons = hull.get_disaggregation_constraint(m.y, m.disjunction)
        self.assertTrue(cons.active)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, m.y - y_y2 - y_y1 == 0.0)

    @unittest.skipUnless(gurobi_available, 'Gurobi is not available')
    def test_do_not_assume_nested_indicators_local(self):
        ct.check_do_not_assume_nested_indicators_local(self, 'gdp.hull')