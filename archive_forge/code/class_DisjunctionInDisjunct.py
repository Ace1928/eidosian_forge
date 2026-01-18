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
class DisjunctionInDisjunct(unittest.TestCase, CommonTests):

    def setUp(self):
        random.seed(666)

    def test_disjuncts_inactive(self):
        ct.check_disjuncts_inactive_nested(self, 'bigm')

    def test_deactivated_disjunct_leaves_nested_disjuncts_active(self):
        ct.check_deactivated_disjunct_leaves_nested_disjunct_active(self, 'bigm')

    def check_disjunction_transformation_block_structure(self, transBlock, pairs):
        self.assertIsInstance(transBlock, Block)
        disjBlock = transBlock.relaxedDisjuncts
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), len(pairs))
        bigm = TransformationFactory('gdp.bigm')
        for i, j in pairs:
            for comp in j:
                self.assertIs(bigm.get_transformed_constraints(comp)[0].parent_block(), disjBlock[i])

    def test_transformation_block_structure(self):
        m = models.makeNestedDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(m)
        transBlock = m.disjunction.algebraic_constraint.parent_block()
        pairs = [(0, [m.simpledisjunct.innerdisjunct0.c]), (1, [m.simpledisjunct.innerdisjunct1.c]), (2, []), (5, [m.disjunct[0].c]), (2, [m.disjunct[1].innerdisjunct[0].c]), (3, [m.disjunct[1].innerdisjunct[1].c]), (6, [])]
        self.check_disjunction_transformation_block_structure(transBlock, pairs)
        self.assertIsInstance(transBlock.component('disjunction_xor'), Constraint)

    def test_mappings_between_disjunctions_and_xors(self):
        m = models.makeNestedDisjunctions()
        transform = TransformationFactory('gdp.bigm')
        transform.apply_to(m)
        transBlock1 = m.component('_pyomo_gdp_bigm_reformulation')
        transBlock2 = m.disjunct[1].component('_pyomo_gdp_bigm_reformulation')
        transBlock3 = m.simpledisjunct.component('_pyomo_gdp_bigm_reformulation')
        disjunctionPairs = [(m.disjunction, transBlock1.disjunction_xor), (m.disjunct[1].innerdisjunction[0], transBlock2.innerdisjunction_xor[0]), (m.simpledisjunct.innerdisjunction, transBlock3.innerdisjunction_xor)]
        for disjunction, xor in disjunctionPairs:
            self.assertIs(disjunction.algebraic_constraint, xor)
            self.assertIs(transform.get_src_disjunction(xor), disjunction)

    def test_disjunct_mappings(self):
        m = models.makeNestedDisjunctions()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        disjunctBlocks = m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts
        self.assertIs(m.disjunct[1].innerdisjunct[0].transformation_block, disjunctBlocks[2])
        self.assertIs(disjunctBlocks[2]._src_disjunct(), m.disjunct[1].innerdisjunct[0])
        self.assertIs(m.disjunct[1].innerdisjunct[1].transformation_block, disjunctBlocks[3])
        self.assertIs(disjunctBlocks[3]._src_disjunct(), m.disjunct[1].innerdisjunct[1])
        self.assertIs(m.simpledisjunct.innerdisjunct0.transformation_block, disjunctBlocks[0])
        self.assertIs(disjunctBlocks[0]._src_disjunct(), m.simpledisjunct.innerdisjunct0)
        self.assertIs(m.simpledisjunct.innerdisjunct1.transformation_block, disjunctBlocks[1])
        self.assertIs(disjunctBlocks[1]._src_disjunct(), m.simpledisjunct.innerdisjunct1)

    def test_m_value_mappings(self):
        m = models.makeNestedDisjunctions()
        bigm = TransformationFactory('gdp.bigm')
        m.simpledisjunct.BigM = Suffix(direction=Suffix.LOCAL)
        m.simpledisjunct.BigM[None] = 58
        m.simpledisjunct.BigM[m.simpledisjunct.innerdisjunct0.c] = 42
        bigms = {m.disjunct[1].innerdisjunct[0]: 89}
        bigm.apply_to(m, bigM=bigms)
        (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.disjunct[1].innerdisjunct[0].c)
        self.assertIs(l_src, bigms)
        self.assertIs(u_src, bigms)
        self.assertIs(l_key, m.disjunct[1].innerdisjunct[0])
        self.assertIs(u_key, m.disjunct[1].innerdisjunct[0])
        self.assertEqual(l_val, -89)
        self.assertEqual(u_val, 89)
        (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.disjunct[1].innerdisjunct[1].c)
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -5)
        self.assertIsNone(u_val)
        (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.disjunct[0].c)
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -11)
        self.assertEqual(u_val, 7)
        (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.disjunct[1].c)
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 21)
        (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.simpledisjunct.innerdisjunct0.c)
        self.assertIsNone(l_src)
        self.assertIs(u_src, m.simpledisjunct.BigM)
        self.assertIsNone(l_key)
        self.assertIs(u_key, m.simpledisjunct.innerdisjunct0.c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 42)
        (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.simpledisjunct.innerdisjunct1.c)
        self.assertIs(l_src, m.simpledisjunct.BigM)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -58)
        self.assertIsNone(u_val)

    def check_bigM_constraint(self, cons, variable, M, indicator_var):
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -M)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, variable, 1)
        ct.check_linear_coef(self, repn, indicator_var, M)

    def check_inner_xor_constraint(self, inner_disjunction, outer_disjunct, bigm):
        inner_xor = inner_disjunction.algebraic_constraint
        sum_indicators = sum((d.binary_indicator_var for d in inner_disjunction.disjuncts))
        assertExpressionsEqual(self, inner_xor.expr, sum_indicators == 1)
        self.assertFalse(inner_xor.active)
        cons = bigm.get_transformed_constraints(inner_xor)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        ct.check_obj_in_active_tree(self, lb)
        lb_expr = self.simplify_cons(lb, leq=False)
        assertExpressionsEqual(self, lb_expr, 1.0 <= sum_indicators - outer_disjunct.binary_indicator_var + 1.0)
        ub = cons[1]
        ct.check_obj_in_active_tree(self, ub)
        ub_expr = self.simplify_cons(ub, leq=True)
        assertExpressionsEqual(self, ub_expr, sum_indicators + outer_disjunct.binary_indicator_var - 1 <= 1.0)

    def test_transformed_constraints(self):
        m = models.makeNestedDisjunctions()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        cons1 = bigm.get_transformed_constraints(m.disjunct[1].innerdisjunct[0].c)
        self.assertEqual(len(cons1), 2)
        cons1lb = cons1[0]
        cons1ub = cons1[1]
        self.assertEqual(cons1lb.lower, 0)
        self.assertIsNone(cons1lb.upper)
        assertExpressionsEqual(self, cons1lb.body, EXPR.SumExpression([m.z, EXPR.NegationExpression((EXPR.ProductExpression((0.0, EXPR.LinearExpression([1, EXPR.MonomialTermExpression((-1, m.disjunct[1].innerdisjunct[0].binary_indicator_var))]))),))]))
        self.assertIsNone(cons1ub.lower)
        self.assertEqual(cons1ub.upper, 0)
        self.check_bigM_constraint(cons1ub, m.z, 10, m.disjunct[1].innerdisjunct[0].indicator_var)
        cons2 = bigm.get_transformed_constraints(m.disjunct[1].innerdisjunct[1].c)
        self.assertEqual(len(cons2), 1)
        cons2lb = cons2[0]
        self.assertEqual(cons2lb.lower, 5)
        self.assertIsNone(cons2lb.upper)
        self.check_bigM_constraint(cons2lb, m.z, -5, m.disjunct[1].innerdisjunct[1].indicator_var)
        cons3 = bigm.get_transformed_constraints(m.simpledisjunct.innerdisjunct0.c)
        self.assertEqual(len(cons3), 1)
        cons3ub = cons3[0]
        self.assertEqual(cons3ub.upper, 2)
        self.assertIsNone(cons3ub.lower)
        self.check_bigM_constraint(cons3ub, m.x, 7, m.simpledisjunct.innerdisjunct0.indicator_var)
        cons4 = bigm.get_transformed_constraints(m.simpledisjunct.innerdisjunct1.c)
        self.assertEqual(len(cons4), 1)
        cons4lb = cons4[0]
        self.assertEqual(cons4lb.lower, 4)
        self.assertIsNone(cons4lb.upper)
        self.check_bigM_constraint(cons4lb, m.x, -13, m.simpledisjunct.innerdisjunct1.indicator_var)
        self.check_inner_xor_constraint(m.simpledisjunct.innerdisjunction, m.simpledisjunct, bigm)
        cons6 = bigm.get_transformed_constraints(m.disjunct[0].c)
        self.assertEqual(len(cons6), 2)
        cons6lb = cons6[0]
        self.assertIsNone(cons6lb.upper)
        self.assertEqual(cons6lb.lower, 2)
        self.check_bigM_constraint(cons6lb, m.x, -11, m.disjunct[0].indicator_var)
        cons6ub = cons6[1]
        self.assertIsNone(cons6ub.lower)
        self.assertEqual(cons6ub.upper, 2)
        self.check_bigM_constraint(cons6ub, m.x, 7, m.disjunct[0].indicator_var)
        self.check_inner_xor_constraint(m.disjunct[1].innerdisjunction[0], m.disjunct[1], bigm)
        cons8 = bigm.get_transformed_constraints(m.disjunct[1].c)
        self.assertEqual(len(cons8), 1)
        cons8ub = cons8[0]
        self.assertIsNone(cons8ub.lower)
        self.assertEqual(cons8ub.upper, 2)
        self.check_bigM_constraint(cons8ub, m.a, 21, m.disjunct[1].indicator_var)

    def test_unique_reference_to_nested_indicator_var(self):
        ct.check_unique_reference_to_nested_indicator_var(self, 'bigm')

    def test_disjunct_targets_inactive(self):
        ct.check_disjunct_targets_inactive(self, 'bigm')

    def test_disjunct_only_targets_transformed(self):
        ct.check_disjunct_only_targets_transformed(self, 'bigm')

    def test_disjunctData_targets_inactive(self):
        ct.check_disjunctData_targets_inactive(self, 'bigm')

    def test_disjunctData_only_targets_transformed(self):
        ct.check_disjunctData_only_targets_transformed(self, 'bigm')

    def test_cannot_call_transformation_on_disjunction(self):
        ct.check_cannot_call_transformation_on_disjunction(self, 'bigm')

    def test_disjunction_target_err(self):
        ct.check_disjunction_target_err(self, 'bigm')

    def test_nested_disjunction_target(self):
        ct.check_nested_disjunction_target(self, 'bigm')

    def test_target_appears_twice(self):
        ct.check_target_appears_twice(self, 'bigm')

    def test_create_using(self):
        m = models.makeNestedDisjunctions()
        self.diff_apply_to_and_create_using(m)

    def test_indexed_nested_disjunction(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d1.indexedDisjunct1 = Disjunct([0, 1])
        m.d1.indexedDisjunct2 = Disjunct([0, 1])

        @m.d1.Disjunction([0, 1])
        def innerIndexed(d, i):
            return [d.indexedDisjunct1[i], d.indexedDisjunct2[i]]
        m.d2 = Disjunct()
        m.outer = Disjunction(expr=[m.d1, m.d2])
        TransformationFactory('gdp.bigm').apply_to(m)
        disjuncts = [m.d1, m.d2, m.d1.indexedDisjunct1[0], m.d1.indexedDisjunct1[1], m.d1.indexedDisjunct2[0], m.d1.indexedDisjunct2[1]]
        for disjunct in disjuncts:
            self.assertIs(disjunct.transformation_block.parent_component(), m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts)

    def check_first_disjunct_constraint(self, disj1c, x, ind_var):
        self.assertEqual(len(disj1c), 1)
        cons = disj1c[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 1)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_quadratic())
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.quadratic_vars), 4)
        ct.check_linear_coef(self, repn, ind_var, 143)
        self.assertEqual(repn.constant, -143)
        for i in range(1, 5):
            ct.check_squared_term_coef(self, repn, x[i], 1)

    def check_second_disjunct_constraint(self, disj2c, x, ind_var):
        self.assertEqual(len(disj2c), 1)
        cons = disj2c[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 1)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_quadratic())
        self.assertEqual(len(repn.linear_vars), 5)
        self.assertEqual(len(repn.quadratic_vars), 4)
        self.assertEqual(repn.constant, -63)
        ct.check_linear_coef(self, repn, ind_var, 99)
        for i in range(1, 5):
            ct.check_squared_term_coef(self, repn, x[i], 1)
            ct.check_linear_coef(self, repn, x[i], -6)

    def simplify_cons(self, cons, leq):
        visitor = LinearRepnVisitor({}, {}, {}, None)
        repn = visitor.walk_expression(cons.body)
        self.assertIsNone(repn.nonlinear)
        if leq:
            self.assertIsNone(cons.lower)
            ub = cons.upper
            return ub >= repn.to_expression(visitor)
        else:
            self.assertIsNone(cons.upper)
            lb = cons.lower
            return lb <= repn.to_expression(visitor)

    def check_hierarchical_nested_model(self, m, bigm):
        outer_xor = m.disjunction_block.disjunction.algebraic_constraint
        ct.check_two_term_disjunction_xor(self, outer_xor, m.disj1, m.disjunct_block.disj2)
        self.check_inner_xor_constraint(m.disjunct_block.disj2.disjunction, m.disjunct_block.disj2, bigm)
        disj1c = bigm.get_transformed_constraints(m.disj1.c)
        self.check_first_disjunct_constraint(disj1c, m.x, m.disj1.binary_indicator_var)
        disj2c = bigm.get_transformed_constraints(m.disjunct_block.disj2.c)
        self.check_second_disjunct_constraint(disj2c, m.x, m.disjunct_block.disj2.binary_indicator_var)
        innerd1c = bigm.get_transformed_constraints(m.disjunct_block.disj2.disjunction_disjuncts[0].constraint[1])
        self.check_first_disjunct_constraint(innerd1c, m.x, m.disjunct_block.disj2.disjunction_disjuncts[0].binary_indicator_var)
        innerd2c = bigm.get_transformed_constraints(m.disjunct_block.disj2.disjunction_disjuncts[1].constraint[1])
        self.check_second_disjunct_constraint(innerd2c, m.x, m.disjunct_block.disj2.disjunction_disjuncts[1].binary_indicator_var)

    def test_hierarchical_badly_ordered_targets(self):
        m = models.makeHierarchicalNested_DeclOrderMatchesInstantiationOrder()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m, targets=[m.disjunction_block, m.disjunct_block.disj2])
        self.check_hierarchical_nested_model(m, bigm)

    def test_decl_order_opposite_instantiation_order(self):
        m = models.makeHierarchicalNested_DeclOrderOppositeInstantiationOrder()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        self.check_hierarchical_nested_model(m, bigm)

    @unittest.skipUnless(gurobi_available, 'Gurobi is not available')
    def test_do_not_assume_nested_indicators_local(self):
        ct.check_do_not_assume_nested_indicators_local(self, 'gdp.bigm')