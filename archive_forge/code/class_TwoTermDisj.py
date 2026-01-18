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
class TwoTermDisj(unittest.TestCase, CommonTests):

    def setUp(self):
        random.seed(666)

    def test_transformation_block(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.hull').apply_to(m)
        transBlock = m._pyomo_gdp_hull_reformulation
        self.assertIsInstance(transBlock, Block)
        lbub = transBlock.lbub
        self.assertIsInstance(lbub, Set)
        self.assertEqual(lbub, ['lb', 'ub', 'eq'])
        disjBlock = transBlock.relaxedDisjuncts
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)

    def test_transformation_block_name_collision(self):
        ct.check_transformation_block_name_collision(self, 'hull')

    def test_disaggregated_vars(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.hull').apply_to(m)
        transBlock = m._pyomo_gdp_hull_reformulation
        disjBlock = transBlock.relaxedDisjuncts
        for i in [0, 1]:
            relaxationBlock = disjBlock[i]
            x = relaxationBlock.disaggregatedVars.x
            if i == 1:
                w = relaxationBlock.disaggregatedVars.w
                y = transBlock._disaggregatedVars[0]
            elif i == 0:
                y = relaxationBlock.disaggregatedVars.y
                w = transBlock._disaggregatedVars[1]
            self.assertIs(w.ctype, Var)
            self.assertIsInstance(x, Var)
            self.assertIs(y.ctype, Var)
            self.assertIsInstance(w.domain, RealSet)
            self.assertIsInstance(x.domain, RealSet)
            self.assertIsInstance(y.domain, RealSet)
            self.assertEqual(w.lb, 0)
            self.assertEqual(w.ub, 7)
            self.assertEqual(x.lb, 0)
            self.assertEqual(x.ub, 8)
            self.assertEqual(y.lb, -10)
            self.assertEqual(y.ub, 0)

    def check_furman_et_al_denominator(self, expr, ind_var):
        self.assertEqual(expr._const, EPS)
        self.assertEqual(len(expr._args), 1)
        self.assertEqual(len(expr._coef), 1)
        self.assertEqual(expr._coef[0], 1 - EPS)
        self.assertIs(expr._args[0], ind_var)

    def test_transformed_constraint_nonlinear(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts
        disj1c = hull.get_transformed_constraints(m.d[0].c)
        self.assertEqual(len(disj1c), 1)
        cons = disj1c[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertFalse(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 1)
        EPS_1 = 1 - EPS
        _disj = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts[0]
        assertExpressionsEqual(self, cons.body, EXPR.SumExpression([EXPR.ProductExpression((EXPR.LinearExpression([EXPR.MonomialTermExpression((EPS_1, m.d[0].binary_indicator_var)), EPS]), EXPR.SumExpression([EXPR.DivisionExpression((_disj.disaggregatedVars.x, EXPR.LinearExpression([EXPR.MonomialTermExpression((EPS_1, m.d[0].binary_indicator_var)), EPS]))), EXPR.PowExpression((EXPR.DivisionExpression((_disj.disaggregatedVars.y, EXPR.LinearExpression([EXPR.MonomialTermExpression((EPS_1, m.d[0].binary_indicator_var)), EPS]))), 2))]))), EXPR.NegationExpression((EXPR.ProductExpression((0.0, EXPR.LinearExpression([1, EXPR.MonomialTermExpression((-1, m.d[0].binary_indicator_var))]))),)), EXPR.MonomialTermExpression((-14.0, m.d[0].binary_indicator_var))]))

    def test_transformed_constraints_linear(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts
        c1 = hull.get_transformed_constraints(m.d[1].c1)
        self.assertEqual(len(c1), 1)
        cons = c1[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.x, -1)
        ct.check_linear_coef(self, repn, m.d[1].indicator_var, 2)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(disjBlock[1].disaggregatedVars.x.lb, 0)
        self.assertEqual(disjBlock[1].disaggregatedVars.x.ub, 8)
        c2 = hull.get_transformed_constraints(m.d[1].c2)
        self.assertEqual(len(c2), 1)
        cons = c2[0]
        self.assertEqual(cons.lower, 0)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.w, 1)
        ct.check_linear_coef(self, repn, m.d[1].indicator_var, -3)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(disjBlock[1].disaggregatedVars.w.lb, 0)
        self.assertEqual(disjBlock[1].disaggregatedVars.w.ub, 7)
        c3 = hull.get_transformed_constraints(m.d[1].c3)
        self.assertEqual(len(c3), 2)
        cons = c3[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.x, -1)
        ct.check_linear_coef(self, repn, m.d[1].indicator_var, 1)
        self.assertEqual(repn.constant, 0)
        cons = c3[1]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.x, 1)
        ct.check_linear_coef(self, repn, m.d[1].indicator_var, -3)
        self.assertEqual(repn.constant, 0)

    def check_bound_constraints_on_disjBlock(self, cons, disvar, indvar, lb, ub):
        self.assertIsInstance(cons, Constraint)
        self.assertEqual(len(cons), 2)
        varlb = cons['lb']
        self.assertIsNone(varlb.lower)
        self.assertEqual(varlb.upper, 0)
        repn = generate_standard_repn(varlb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, indvar, lb)
        ct.check_linear_coef(self, repn, disvar, -1)
        varub = cons['ub']
        self.assertIsNone(varub.lower)
        self.assertEqual(varub.upper, 0)
        repn = generate_standard_repn(varub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, indvar, -ub)
        ct.check_linear_coef(self, repn, disvar, 1)

    def check_bound_constraints_on_disjunctionBlock(self, varlb, varub, disvar, indvar, lb, ub):
        self.assertIsNone(varlb.lower)
        self.assertEqual(varlb.upper, 0)
        repn = generate_standard_repn(varlb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, lb)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, indvar, -lb)
        ct.check_linear_coef(self, repn, disvar, -1)
        self.assertIsNone(varub.lower)
        self.assertEqual(varub.upper, 0)
        repn = generate_standard_repn(varub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -ub)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, indvar, ub)
        ct.check_linear_coef(self, repn, disvar, 1)

    def test_disaggregatedVar_bounds(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.hull').apply_to(m)
        transBlock = m._pyomo_gdp_hull_reformulation
        disjBlock = transBlock.relaxedDisjuncts
        for i in [0, 1]:
            self.check_bound_constraints_on_disjBlock(disjBlock[i].x_bounds, disjBlock[i].disaggregatedVars.x, m.d[i].indicator_var, 1, 8)
            if i == 1:
                self.check_bound_constraints_on_disjBlock(disjBlock[i].w_bounds, disjBlock[i].disaggregatedVars.w, m.d[i].indicator_var, 2, 7)
                self.check_bound_constraints_on_disjunctionBlock(transBlock._boundsConstraints[0, 'lb'], transBlock._boundsConstraints[0, 'ub'], transBlock._disaggregatedVars[0], m.d[0].indicator_var, -10, -3)
            elif i == 0:
                self.check_bound_constraints_on_disjBlock(disjBlock[i].y_bounds, disjBlock[i].disaggregatedVars.y, m.d[i].indicator_var, -10, -3)
                self.check_bound_constraints_on_disjunctionBlock(transBlock._boundsConstraints[1, 'lb'], transBlock._boundsConstraints[1, 'ub'], transBlock._disaggregatedVars[1], m.d[1].indicator_var, 2, 7)

    def test_error_for_or(self):
        m = models.makeTwoTermDisj_Nonlinear()
        m.disjunction.xor = False
        self.assertRaisesRegex(GDP_Error, "Cannot do hull reformulation for Disjunction 'disjunction' with OR constraint. Must be an XOR!*", TransformationFactory('gdp.hull').apply_to, m)

    def check_disaggregation_constraint(self, cons, var, disvar1, disvar2):
        assertExpressionsEqual(self, cons.expr, var == disvar1 + disvar2)

    def test_disaggregation_constraint(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        transBlock = m._pyomo_gdp_hull_reformulation
        disjBlock = transBlock.relaxedDisjuncts
        self.check_disaggregation_constraint(hull.get_disaggregation_constraint(m.w, m.disjunction), m.w, transBlock._disaggregatedVars[1], disjBlock[1].disaggregatedVars.w)
        self.check_disaggregation_constraint(hull.get_disaggregation_constraint(m.x, m.disjunction), m.x, disjBlock[0].disaggregatedVars.x, disjBlock[1].disaggregatedVars.x)
        self.check_disaggregation_constraint(hull.get_disaggregation_constraint(m.y, m.disjunction), m.y, transBlock._disaggregatedVars[0], disjBlock[0].disaggregatedVars.y)

    def test_xor_constraint_mapping(self):
        ct.check_xor_constraint_mapping(self, 'hull')

    def test_xor_constraint_mapping_two_disjunctions(self):
        ct.check_xor_constraint_mapping_two_disjunctions(self, 'hull')

    def test_transformed_disjunct_mappings(self):
        ct.check_disjunct_mapping(self, 'hull')

    def test_transformed_constraint_mappings(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts
        orig1 = m.d[0].c
        cons = hull.get_transformed_constraints(orig1)
        self.assertEqual(len(cons), 1)
        trans1 = cons[0]
        self.assertIs(trans1.parent_block(), disjBlock[0])
        self.assertIs(hull.get_src_constraint(trans1), orig1)
        orig1 = m.d[1].c1
        cons = hull.get_transformed_constraints(orig1)
        self.assertEqual(len(cons), 1)
        trans1 = cons[0]
        self.assertIs(trans1.parent_block(), disjBlock[1])
        self.assertIs(hull.get_src_constraint(trans1), orig1)
        orig2 = m.d[1].c2
        cons = hull.get_transformed_constraints(orig2)
        self.assertEqual(len(cons), 1)
        trans2 = cons[0]
        self.assertIs(trans1.parent_block(), disjBlock[1])
        self.assertIs(hull.get_src_constraint(trans2), orig2)
        orig3 = m.d[1].c3
        cons = hull.get_transformed_constraints(orig3)
        self.assertEqual(len(cons), 2)
        trans3 = cons[0]
        self.assertIs(hull.get_src_constraint(trans3), orig3)
        self.assertIs(trans3.parent_block(), disjBlock[1])
        trans32 = cons[1]
        self.assertIs(hull.get_src_constraint(trans32), orig3)
        self.assertIs(trans32.parent_block(), disjBlock[1])

    def test_disaggregatedVar_mappings(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        transBlock = m._pyomo_gdp_hull_reformulation
        disjBlock = transBlock.relaxedDisjuncts
        for i in [0, 1]:
            mappings = ComponentMap()
            mappings[m.x] = disjBlock[i].disaggregatedVars.x
            if i == 1:
                mappings[m.w] = disjBlock[i].disaggregatedVars.w
                mappings[m.y] = transBlock._disaggregatedVars[0]
            elif i == 0:
                mappings[m.y] = disjBlock[i].disaggregatedVars.y
                mappings[m.w] = transBlock._disaggregatedVars[1]
            for orig, disagg in mappings.items():
                self.assertIs(hull.get_src_var(disagg), orig)
                self.assertIs(hull.get_disaggregated_var(orig, m.d[i]), disagg)

    def test_bigMConstraint_mappings(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        transBlock = m._pyomo_gdp_hull_reformulation
        disjBlock = transBlock.relaxedDisjuncts
        for i in [0, 1]:
            mappings = ComponentMap()
            mappings[disjBlock[i].disaggregatedVars.x] = disjBlock[i].x_bounds
            if i == 1:
                mappings[disjBlock[i].disaggregatedVars.w] = disjBlock[i].w_bounds
                mappings[transBlock._disaggregatedVars[0]] = Reference(transBlock._boundsConstraints[0, ...])
            elif i == 0:
                mappings[disjBlock[i].disaggregatedVars.y] = disjBlock[i].y_bounds
                mappings[transBlock._disaggregatedVars[1]] = Reference(transBlock._boundsConstraints[1, ...])
            for var, cons in mappings.items():
                returned_cons = hull.get_var_bounds_constraint(var)
                for key, constraintData in cons.items():
                    self.assertIs(returned_cons[key], constraintData)

    def test_create_using_nonlinear(self):
        m = models.makeTwoTermDisj_Nonlinear()
        self.diff_apply_to_and_create_using(m)

    def test_locally_declared_var_bounds_used_globally(self):
        m = models.localVar()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        y_disagg = m.disj2.transformation_block.disaggregatedVars.component('disj2.y')
        cons = hull.get_var_bounds_constraint(y_disagg)
        lb = cons['lb']
        self.assertIsNone(lb.lower)
        self.assertEqual(value(lb.upper), 0)
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        ct.check_linear_coef(self, repn, m.disj2.indicator_var, 1)
        ct.check_linear_coef(self, repn, y_disagg, -1)
        ub = cons['ub']
        self.assertIsNone(ub.lower)
        self.assertEqual(value(ub.upper), 0)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        ct.check_linear_coef(self, repn, y_disagg, 1)
        ct.check_linear_coef(self, repn, m.disj2.indicator_var, -3)

    def test_locally_declared_variables_disaggregated(self):
        m = models.localVar()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        disj1y = hull.get_disaggregated_var(m.disj2.y, m.disj1)
        disj2y = hull.get_disaggregated_var(m.disj2.y, m.disj2)
        self.assertIs(disj1y, m.disj1.transformation_block.parent_block()._disaggregatedVars[0])
        self.assertIs(disj2y, m.disj2.transformation_block.disaggregatedVars.component('disj2.y'))
        self.assertIs(hull.get_src_var(disj1y), m.disj2.y)
        self.assertIs(hull.get_src_var(disj2y), m.disj2.y)

    def test_global_vars_local_to_a_disjunction_disaggregated(self):
        m = ConcreteModel()
        m.disj1 = Disjunct()
        m.disj1.x = Var(bounds=(1, 10))
        m.disj1.y = Var(bounds=(2, 11))
        m.disj1.cons1 = Constraint(expr=m.disj1.x + m.disj1.y <= 5)
        m.disj2 = Disjunct()
        m.disj2.cons = Constraint(expr=m.disj1.y >= 8)
        m.disjunction1 = Disjunction(expr=[m.disj1, m.disj2])
        m.disj3 = Disjunct()
        m.disj3.cons = Constraint(expr=m.disj1.x >= 7)
        m.disj4 = Disjunct()
        m.disj4.cons = Constraint(expr=m.disj1.y == 3)
        m.disjunction2 = Disjunction(expr=[m.disj3, m.disj4])
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        disj = m.disj1
        transBlock = disj.transformation_block
        varBlock = transBlock.disaggregatedVars
        self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 2)
        x = varBlock.component('disj1.x')
        y = varBlock.component('disj1.y')
        self.assertIsInstance(x, Var)
        self.assertIsInstance(y, Var)
        self.assertIs(hull.get_disaggregated_var(m.disj1.x, disj), x)
        self.assertIs(hull.get_src_var(x), m.disj1.x)
        self.assertIs(hull.get_disaggregated_var(m.disj1.y, disj), y)
        self.assertIs(hull.get_src_var(y), m.disj1.y)
        for disj in [m.disj2, m.disj4]:
            transBlock = disj.transformation_block
            varBlock = transBlock.disaggregatedVars
            self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 1)
            y = varBlock.component('disj1.y')
            self.assertIsInstance(y, Var)
            self.assertIs(hull.get_disaggregated_var(m.disj1.y, disj), y)
            self.assertIs(hull.get_src_var(y), m.disj1.y)
        disj = m.disj3
        transBlock = disj.transformation_block
        varBlock = transBlock.disaggregatedVars
        self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 1)
        x = varBlock.component('disj1.x')
        self.assertIsInstance(x, Var)
        self.assertIs(hull.get_disaggregated_var(m.disj1.x, disj), x)
        self.assertIs(hull.get_src_var(x), m.disj1.x)
        x2 = m.disjunction1.algebraic_constraint.parent_block()._disaggregatedVars[0]
        self.assertIs(hull.get_disaggregated_var(m.disj1.x, m.disj2), x2)
        self.assertIs(hull.get_src_var(x2), m.disj1.x)
        agg_cons = hull.get_disaggregation_constraint(m.disj1.x, m.disjunction1)
        assertExpressionsEqual(self, agg_cons.expr, m.disj1.x == x2 + hull.get_disaggregated_var(m.disj1.x, m.disj1))
        x2 = m.disjunction2.algebraic_constraint.parent_block()._disaggregatedVars[1]
        y1 = m.disjunction2.algebraic_constraint.parent_block()._disaggregatedVars[2]
        self.assertIs(hull.get_disaggregated_var(m.disj1.x, m.disj4), x2)
        self.assertIs(hull.get_src_var(x2), m.disj1.x)
        self.assertIs(hull.get_disaggregated_var(m.disj1.y, m.disj3), y1)
        self.assertIs(hull.get_src_var(y1), m.disj1.y)
        agg_cons = hull.get_disaggregation_constraint(m.disj1.x, m.disjunction2)
        assertExpressionsEqual(self, agg_cons.expr, m.disj1.x == x2 + hull.get_disaggregated_var(m.disj1.x, m.disj3))
        agg_cons = hull.get_disaggregation_constraint(m.disj1.y, m.disjunction2)
        assertExpressionsEqual(self, agg_cons.expr, m.disj1.y == y1 + hull.get_disaggregated_var(m.disj1.y, m.disj4))

    def check_name_collision_disaggregated_vars(self, m, disj):
        hull = TransformationFactory('gdp.hull')
        transBlock = disj.transformation_block
        varBlock = transBlock.disaggregatedVars
        self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 2)
        x2 = varBlock.component("'disj1.x'")
        x = varBlock.component('disj1.x')
        x_orig = m.component('disj1.x')
        self.assertIsInstance(x, Var)
        self.assertIsInstance(x2, Var)
        self.assertIs(hull.get_disaggregated_var(m.disj1.x, disj), x)
        self.assertIs(hull.get_src_var(x), m.disj1.x)
        self.assertIs(hull.get_disaggregated_var(x_orig, disj), x2)
        self.assertIs(hull.get_src_var(x2), x_orig)

    def test_disaggregated_var_name_collision(self):
        m = ConcreteModel()
        x = Var(bounds=(2, 11))
        m.add_component('disj1.x', x)
        m.disj1 = Disjunct()
        m.disj1.x = Var(bounds=(1, 10))
        m.disj1.cons1 = Constraint(expr=m.disj1.x + x <= 5)
        m.disj2 = Disjunct()
        m.disj2.cons = Constraint(expr=x >= 8)
        m.disj2.cons1 = Constraint(expr=m.disj1.x == 3)
        m.disjunction1 = Disjunction(expr=[m.disj1, m.disj2])
        m.disj3 = Disjunct()
        m.disj3.cons = Constraint(expr=m.disj1.x >= 7)
        m.disj3.cons1 = Constraint(expr=x >= 10)
        m.disj4 = Disjunct()
        m.disj4.cons = Constraint(expr=x == 3)
        m.disj4.cons1 = Constraint(expr=m.disj1.x == 4)
        m.disjunction2 = Disjunction(expr=[m.disj3, m.disj4])
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        for disj in (m.disj1, m.disj2, m.disj3, m.disj4):
            self.check_name_collision_disaggregated_vars(m, disj)

    def test_do_not_transform_user_deactivated_disjuncts(self):
        ct.check_user_deactivated_disjuncts(self, 'hull')

    def test_improperly_deactivated_disjuncts(self):
        ct.check_improperly_deactivated_disjuncts(self, 'hull')

    def test_do_not_transform_userDeactivated_IndexedDisjunction(self):
        ct.check_do_not_transform_userDeactivated_indexedDisjunction(self, 'hull')

    def test_disjunction_deactivated(self):
        ct.check_disjunction_deactivated(self, 'hull')

    def test_disjunctDatas_deactivated(self):
        ct.check_disjunctDatas_deactivated(self, 'hull')

    def test_deactivated_constraints(self):
        ct.check_deactivated_constraints(self, 'hull')

    def check_no_double_transformation(self):
        ct.check_do_not_transform_twice_if_disjunction_reactivated(self, 'hull')

    def test_indicator_vars(self):
        ct.check_indicator_vars(self, 'hull')

    def test_xor_constraints(self):
        ct.check_xor_constraint(self, 'hull')

    def test_unbounded_var_error(self):
        m = models.makeTwoTermDisj_Nonlinear()
        m.w.setlb(None)
        m.w.setub(None)
        self.assertRaisesRegex(GDP_Error, 'Variables that appear in disjuncts must be bounded in order to use the hull transformation! Missing bound for w.*', TransformationFactory('gdp.hull').apply_to, m)

    def check_threeTermDisj_IndexedConstraints(self, m, lb):
        transBlock = m._pyomo_gdp_hull_reformulation
        hull = TransformationFactory('gdp.hull')
        self.assertEqual(len(list(m.component_objects(Block, descend_into=False))), 1)
        self.assertEqual(len(list(m.component_objects(Disjunct))), 1)
        for i in [1, 2, 3]:
            relaxed = transBlock.relaxedDisjuncts[i - 1]
            self.assertEqual(len(list(relaxed.disaggregatedVars.component_objects(Var))), i)
            self.assertEqual(len(list(relaxed.disaggregatedVars.component_data_objects(Var))), i)
            self.assertEqual(len(list(relaxed.component_objects(Constraint))), 1 + i)
            if lb == 0:
                self.assertEqual(len(list(relaxed.component_data_objects(Constraint))), i + i)
            else:
                self.assertEqual(len(list(relaxed.component_data_objects(Constraint))), 2 * i + i)
            for j in range(1, i + 1):
                cons = hull.get_transformed_constraints(m.d[i].c[j])
                self.assertEqual(len(cons), 1)
                self.assertIs(cons[0].parent_block(), relaxed)
        self.assertEqual(len(list(transBlock.component_objects(Var, descend_into=False))), 1)
        self.assertEqual(len(list(transBlock.component_data_objects(Var, descend_into=False))), 2)
        self.assertEqual(len(list(transBlock.component_objects(Constraint, descend_into=False))), 3)
        if lb == 0:
            self.assertEqual(len(list(transBlock.component_data_objects(Constraint, descend_into=False))), 6)
        else:
            self.assertEqual(len(list(transBlock.component_data_objects(Constraint, descend_into=False))), 8)

    def test_indexed_constraints_in_disjunct(self):
        m = models.makeThreeTermDisj_IndexedConstraints()
        TransformationFactory('gdp.hull').apply_to(m)
        self.check_threeTermDisj_IndexedConstraints(m, lb=0)

    def test_virtual_indexed_constraints_in_disjunct(self):
        m = ConcreteModel()
        m.I = [1, 2, 3]
        m.x = Var(m.I, bounds=(-1, 10))

        def d_rule(d, j):
            m = d.model()
            d.c = Constraint(Any)
            for k in range(j):
                d.c[k + 1] = m.x[k + 1] >= k + 1
        m.d = Disjunct(m.I, rule=d_rule)
        m.disjunction = Disjunction(expr=[m.d[i] for i in m.I])
        TransformationFactory('gdp.hull').apply_to(m)
        self.check_threeTermDisj_IndexedConstraints(m, lb=-1)

    def test_do_not_transform_deactivated_constraintDatas(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.a[1].setlb(0)
        m.a[1].setub(100)
        m.a[2].setlb(0)
        m.a[2].setub(100)
        m.b.simpledisj1.c[1].deactivate()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp', logging.ERROR):
            self.assertRaisesRegex(KeyError, '.*b.simpledisj1.c\\[1\\]', hull.get_transformed_constraints, m.b.simpledisj1.c[1])
        self.assertRegex(log.getvalue(), ".*Constraint 'b.simpledisj1.c\\[1\\]' has not been transformed.")
        transformed = hull.get_transformed_constraints(m.b.simpledisj1.c[2])
        self.assertEqual(len(transformed), 1)
        disaggregated_a2 = hull.get_disaggregated_var(m.a[2], m.b.simpledisj1)
        self.assertIs(transformed[0], disaggregated_a2)
        self.assertIsInstance(disaggregated_a2, Var)
        self.assertTrue(disaggregated_a2.is_fixed())
        self.assertEqual(value(disaggregated_a2), 0)
        transformed = hull.get_transformed_constraints(m.b.simpledisj2.c[1])
        self.assertEqual(len(transformed), 1)
        self.assertIs(transformed[0].parent_block(), m.b.simpledisj2.transformation_block)
        transformed = hull.get_transformed_constraints(m.b.simpledisj2.c[2])
        self.assertEqual(len(transformed), 1)
        self.assertIs(transformed[0].parent_block(), m.b.simpledisj2.transformation_block)