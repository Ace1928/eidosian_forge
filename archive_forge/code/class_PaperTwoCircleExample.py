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
class PaperTwoCircleExample(unittest.TestCase, CommonTests):

    def check_disj_constraint(self, c1, upper, auxVar1, auxVar2):
        self.assertIsNone(c1.lower)
        self.assertEqual(value(c1.upper), upper)
        repn = generate_standard_repn(c1.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, 0)
        self.assertIs(repn.linear_vars[0], auxVar1)
        self.assertIs(repn.linear_vars[1], auxVar2)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertEqual(repn.linear_coefs[1], 1)

    def check_global_constraint_disj1(self, c1, auxVar, var1, var2):
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.quadratic_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertIs(repn.linear_vars[0], auxVar)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertEqual(repn.quadratic_coefs[1], 1)
        self.assertIs(repn.quadratic_vars[0][0], var1)
        self.assertIs(repn.quadratic_vars[0][1], var1)
        self.assertIs(repn.quadratic_vars[1][0], var2)
        self.assertIs(repn.quadratic_vars[1][1], var2)
        self.assertIsNone(repn.nonlinear_expr)

    def check_global_constraint_disj2(self, c1, auxVar, var1, var2):
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 3)
        self.assertEqual(len(repn.quadratic_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertEqual(repn.linear_coefs[1], -6)
        self.assertEqual(repn.linear_coefs[2], -1)
        self.assertIs(repn.linear_vars[0], var1)
        self.assertIs(repn.linear_vars[1], var2)
        self.assertIs(repn.linear_vars[2], auxVar)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertEqual(repn.quadratic_coefs[1], 1)
        self.assertIs(repn.quadratic_vars[0][0], var1)
        self.assertIs(repn.quadratic_vars[0][1], var1)
        self.assertIs(repn.quadratic_vars[1][0], var2)
        self.assertIs(repn.quadratic_vars[1][1], var2)
        self.assertIsNone(repn.nonlinear_expr)

    def check_aux_var_bounds(self, aux_vars1, aux_vars2, aux11lb, aux11ub, aux12lb, aux12ub, aux21lb, aux21ub, aux22lb, aux22ub):
        self.assertEqual(len(aux_vars1), 2)
        self.assertAlmostEqual(aux_vars1[0].lb, aux11lb, places=6)
        self.assertAlmostEqual(aux_vars1[0].ub, aux11ub, places=6)
        self.assertAlmostEqual(aux_vars1[1].lb, aux12lb, places=6)
        self.assertAlmostEqual(aux_vars1[1].ub, aux12ub, places=6)
        self.assertAlmostEqual(len(aux_vars2), 2)
        self.assertAlmostEqual(aux_vars2[0].lb, aux21lb, places=6)
        self.assertAlmostEqual(aux_vars2[0].ub, aux21ub, places=6)
        self.assertAlmostEqual(aux_vars2[1].lb, aux22lb, places=6)
        self.assertAlmostEqual(aux_vars2[1].ub, aux22ub, places=6)

    def check_transformation_block_disjuncts_and_constraints(self, m, original_disjunction, disjunction_name=None):
        b = m.component('_pyomo_gdp_partition_disjuncts_reformulation')
        self.assertIsInstance(b, Block)
        self.assertEqual(len(b.component_map(Disjunction)), 1)
        self.assertEqual(len(b.component_map(Disjunct)), 2)
        self.assertEqual(len(b.component_map(Constraint)), 2)
        self.assertEqual(len(b.component_map(LogicalConstraint)), 1)
        if disjunction_name is None:
            disjunction = b.disjunction
        else:
            disjunction = b.component(disjunction_name)
        self.assertEqual(len(disjunction.disjuncts), 2)
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        equivalence = b.component('indicator_var_equalities')
        self.assertIsInstance(equivalence, LogicalConstraint)
        self.assertEqual(len(equivalence), 2)
        for i, variables in enumerate([(original_disjunction.disjuncts[0].indicator_var, disj1.indicator_var), (original_disjunction.disjuncts[1].indicator_var, disj2.indicator_var)]):
            cons = equivalence[i]
            self.assertIsInstance(cons.body, EquivalenceExpression)
            self.assertIs(cons.body.args[0], variables[0])
            self.assertIs(cons.body.args[1], variables[1])
        return (b, disj1, disj2)

    def check_transformation_block_structure(self, m, aux11lb, aux11ub, aux12lb, aux12ub, aux21lb, aux21ub, aux22lb, aux22ub):
        b, disj1, disj2 = self.check_transformation_block_disjuncts_and_constraints(m, m.disjunction)
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)
        aux_vars1 = disj1.component('disjunction_disjuncts[0].constraint[1]_aux_vars')
        aux_vars2 = disj2.component('disjunction_disjuncts[1].constraint[1]_aux_vars')
        self.check_aux_var_bounds(aux_vars1, aux_vars2, aux11lb, aux11ub, aux12lb, aux12ub, aux21lb, aux21ub, aux22lb, aux22ub)
        return (b, disj1, disj2, aux_vars1, aux_vars2)

    def check_disjunct_constraints(self, disj1, disj2, aux_vars1, aux_vars2):
        c = disj1.component('disjunction_disjuncts[0].constraint[1]')
        self.assertEqual(len(c), 1)
        c1 = c[0]
        self.check_disj_constraint(c1, 1, aux_vars1[0], aux_vars1[1])
        c = disj2.component('disjunction_disjuncts[1].constraint[1]')
        self.assertEqual(len(c), 1)
        c2 = c[0]
        self.check_disj_constraint(c2, -35, aux_vars2[0], aux_vars2[1])

    def check_transformation_block(self, m, aux11lb, aux11ub, aux12lb, aux12ub, aux21lb, aux21ub, aux22lb, aux22ub, partitions):
        b, disj1, disj2, aux_vars1, aux_vars2 = self.check_transformation_block_structure(m, aux11lb, aux11ub, aux12lb, aux12ub, aux21lb, aux21ub, aux22lb, aux22ub)
        self.check_disjunct_constraints(disj1, disj2, aux_vars1, aux_vars2)
        c = b.component('disjunction_disjuncts[0].constraint[1]_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], partitions[0][0], partitions[0][1])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], partitions[1][0], partitions[1][1])
        c = b.component('disjunction_disjuncts[1].constraint[1]_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], partitions[0][0], partitions[0][1])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], partitions[1][0], partitions[1][1])

    def test_transformation_block_fbbt_bounds(self):
        m = models.makeBetweenStepsPaperExample()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_fbbt_bounds)
        self.check_transformation_block(m, 0, 72, 0, 72, -72, 96, -72, 96, partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]])

    def check_transformation_block_indexed_var_on_disjunct(self, m, original_disjunction):
        b, disj1, disj2 = self.check_transformation_block_disjuncts_and_constraints(m, original_disjunction)
        self.assertEqual(len(disj1.component_map(Var)), 4)
        self.assertEqual(len(disj2.component_map(Var)), 3)
        aux_vars1 = disj1.component('disj1.c_aux_vars')
        aux_vars2 = disj2.component('disj2.c_aux_vars')
        self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -72, 96, -72, 96)
        c = disj1.component('disj1.c')
        self.assertEqual(len(c), 1)
        c1 = c[0]
        self.check_disj_constraint(c1, 1, aux_vars1[0], aux_vars1[1])
        c = disj2.component('disj2.c')
        self.assertEqual(len(c), 1)
        c2 = c[0]
        self.check_disj_constraint(c2, -35, aux_vars2[0], aux_vars2[1])
        c = b.component('disj1.c_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], m.disj1.x[1], m.disj1.x[2])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], m.disj1.x[3], m.disj1.x[4])
        c = b.component('disj2.c_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], m.disj1.x[1], m.disj1.x[2])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], m.disj1.x[3], m.disj1.x[4])
        return (b, disj1, disj2)

    def test_transformation_block_indexed_var_on_disjunct(self):
        m = models.makeBetweenStepsPaperExample_DeclareVarOnDisjunct()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.disj1.x[1], m.disj1.x[2]], [m.disj1.x[3], m.disj1.x[4]]], compute_bounds_method=compute_fbbt_bounds)
        self.check_transformation_block_indexed_var_on_disjunct(m, m.disjunction)

    def check_transformation_block_nested_disjunction(self, m, disj2, x, disjunction_block=None):
        if disjunction_block is None:
            block_prefix = ''
            disjunction_parent = m
        else:
            block_prefix = disjunction_block + '.'
            disjunction_parent = m.component(disjunction_block)
        inner_b, inner_disj1, inner_disj2 = self.check_transformation_block_disjuncts_and_constraints(disj2, disjunction_parent.disj2.disjunction, '%sdisj2.disjunction' % block_prefix)
        self.assertEqual(len(inner_disj1.component_map(Var)), 3)
        self.assertEqual(len(inner_disj2.component_map(Var)), 3)
        aux_vars1 = inner_disj1.component('%sdisj2.disjunction_disjuncts[0].constraint[1]_aux_vars' % block_prefix)
        aux_vars2 = inner_disj2.component('%sdisj2.disjunction_disjuncts[1].constraint[1]_aux_vars' % block_prefix)
        self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -72, 96, -72, 96)
        c = inner_disj1.component('%sdisj2.disjunction_disjuncts[0].constraint[1]' % block_prefix)
        self.assertEqual(len(c), 1)
        c1 = c[0]
        self.check_disj_constraint(c1, 1, aux_vars1[0], aux_vars1[1])
        c = inner_disj2.component('%sdisj2.disjunction_disjuncts[1].constraint[1]' % block_prefix)
        self.assertEqual(len(c), 1)
        c2 = c[0]
        self.check_disj_constraint(c2, -35, aux_vars2[0], aux_vars2[1])
        c = inner_b.component('%sdisj2.disjunction_disjuncts[0].constraint[1]_split_constraints' % block_prefix)
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], x[1], x[2])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], x[3], x[4])
        c = inner_b.component('%sdisj2.disjunction_disjuncts[1].constraint[1]_split_constraints' % block_prefix)
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], x[1], x[2])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], x[3], x[4])

    def test_transformation_block_nested_disjunction(self):
        m = models.makeBetweenStepsPaperExample_Nested()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.disj1.x[1], m.disj1.x[2]], [m.disj1.x[3], m.disj1.x[4]]], compute_bounds_method=compute_fbbt_bounds)
        b, disj1, disj2 = self.check_transformation_block_indexed_var_on_disjunct(m, m.disjunction)
        self.check_transformation_block_nested_disjunction(m, disj2, m.disj1.x)

    def test_transformation_block_nested_disjunction_outer_disjunction_target(self):
        """We should get identical behavior to the previous test if we
        specify the outer disjunction as the target"""
        m = models.makeBetweenStepsPaperExample_Nested()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, targets=m.disjunction, variable_partitions=[[m.disj1.x[1], m.disj1.x[2]], [m.disj1.x[3], m.disj1.x[4]]], compute_bounds_method=compute_fbbt_bounds)
        b, disj1, disj2 = self.check_transformation_block_indexed_var_on_disjunct(m, m.disjunction)
        self.check_transformation_block_nested_disjunction(m, disj2, m.disj1.x)

    def test_transformation_block_nested_disjunction_badly_ordered_targets(self):
        """This tests that we preprocess targets correctly because we don't
        want to double transform the inner disjunct, which is what would happen
        if we did things in the order given."""
        m = models.makeBetweenStepsPaperExample_Nested()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, targets=[m.disj2, m.disjunction], variable_partitions=[[m.disj1.x[1], m.disj1.x[2]], [m.disj1.x[3], m.disj1.x[4]]], compute_bounds_method=compute_fbbt_bounds)
        b, disj1, disj2 = self.check_transformation_block_indexed_var_on_disjunct(m, m.disjunction)
        self.check_transformation_block_nested_disjunction(m, disj2, m.disj1.x)

    def check_hierarchical_nested_model(self, m):
        b, disj1, disj2 = self.check_transformation_block_disjuncts_and_constraints(m.disjunction_block, m.disjunction_block.disjunction, 'disjunction_block.disjunction')
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)
        aux_vars1 = disj1.component('disj1.c_aux_vars')
        aux_vars2 = disj2.component('disjunct_block.disj2.c_aux_vars')
        self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -72, 96, -72, 96)
        c = disj1.component('disj1.c')
        self.assertEqual(len(c), 1)
        c1 = c[0]
        self.check_disj_constraint(c1, 1, aux_vars1[0], aux_vars1[1])
        c = disj2.component('disjunct_block.disj2.c')
        self.assertEqual(len(c), 1)
        c2 = c[0]
        self.check_disj_constraint(c2, -35, aux_vars2[0], aux_vars2[1])
        c = b.component('disj1.c_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], m.x[3], m.x[4])
        c = b.component('disjunct_block.disj2.c_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], m.x[3], m.x[4])
        self.check_transformation_block_nested_disjunction(m, disj2, m.x, 'disjunct_block')

    def test_hierarchical_nested_badly_ordered_targets(self):
        m = models.makeHierarchicalNested_DeclOrderMatchesInstantiationOrder()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, targets=[m.disjunction_block, m.disjunct_block.disj2], variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_fbbt_bounds)
        self.check_hierarchical_nested_model(m)

    def test_hierarchical_nested_decl_order_opposite_instantiation_order(self):
        m = models.makeHierarchicalNested_DeclOrderOppositeInstantiationOrder()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_fbbt_bounds)
        self.check_hierarchical_nested_model(m)

    def test_transformation_block_nested_disjunction_target(self):
        m = models.makeBetweenStepsPaperExample_Nested()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, targets=m.disj2.disjunction, variable_partitions=[[m.disj1.x[1], m.disj1.x[2]], [m.disj1.x[3], m.disj1.x[4]]], compute_bounds_method=compute_fbbt_bounds)
        self.check_transformation_block_nested_disjunction(m, m.disj2, m.disj1.x)

    @unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
    def test_transformation_block_optimized_bounds(self):
        m = models.makeBetweenStepsPaperExample()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_solver=SolverFactory('gurobi_direct'), compute_bounds_method=compute_optimal_bounds)
        self.check_transformation_block(m, 0, 72, 0, 72, -18, 32, -18, 32, partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]])

    def test_no_solver_error(self):
        m = models.makeBetweenStepsPaperExample()
        with self.assertRaisesRegex(GDP_Error, "No solver was specified to optimize the subproblems for computing expression bounds! Please specify a configured solver in the 'compute_bounds_solver' argument if using 'compute_optimal_bounds.'"):
            TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_optimal_bounds)

    @unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
    def test_transformation_block_better_bounds_in_global_constraints(self):
        m = models.makeBetweenStepsPaperExample()
        m.c1 = Constraint(expr=m.x[1] ** 2 + m.x[2] ** 2 <= 32)
        m.c2 = Constraint(expr=m.x[3] ** 2 + m.x[4] ** 2 <= 32)
        m.c3 = Constraint(expr=(3 - m.x[1]) ** 2 + (3 - m.x[2]) ** 2 <= 32)
        m.c4 = Constraint(expr=(3 - m.x[3]) ** 2 + (3 - m.x[4]) ** 2 <= 32)
        opt = SolverFactory('gurobi_direct')
        opt.options['NonConvex'] = 2
        opt.options['FeasibilityTol'] = 1e-08
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_solver=opt, compute_bounds_method=compute_optimal_bounds)
        self.check_transformation_block(m, 0, 32, 0, 32, -18, 14, -18, 14, partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]])

    @unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
    def test_transformation_block_arbitrary_even_partition(self):
        m = models.makeBetweenStepsPaperExample()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, num_partitions=2, compute_bounds_solver=SolverFactory('gurobi_direct'), compute_bounds_method=compute_optimal_bounds)
        self.check_transformation_block(m, 0, 72, 0, 72, -18, 32, -18, 32, partitions=[[m.x[1], m.x[3]], [m.x[2], m.x[4]]])

    @unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
    def test_assume_fixed_vars_not_permanent(self):
        m = models.makeBetweenStepsPaperExample()
        m.x[1].fix(0)
        m.disjunction.disjuncts[0].indicator_var.fix(True)
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], assume_fixed_vars_permanent=False, compute_bounds_solver=SolverFactory('gurobi_direct'), compute_bounds_method=compute_optimal_bounds)
        self.assertTrue(m.x[1].fixed)
        self.assertEqual(value(m.x[1]), 0)
        self.assertTrue(m.disjunction_disjuncts[0].indicator_var.fixed)
        self.assertTrue(value(m.disjunction.disjuncts[0].indicator_var))
        m.x[1].fixed = False
        self.check_transformation_block(m, 0, 72, 0, 72, -18, 32, -18, 32, partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]])

    @unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
    def test_assume_fixed_vars_permanent(self):
        m = models.makeBetweenStepsPaperExample()
        m.x[1].fix(0)
        m.disjunction.disjuncts[0].indicator_var.fix(True)
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], assume_fixed_vars_permanent=True, compute_bounds_solver=SolverFactory('gurobi_direct'), compute_bounds_method=compute_optimal_bounds)
        self.assertTrue(m.disjunction_disjuncts[0].indicator_var.fixed)
        self.assertTrue(value(m.disjunction.disjuncts[0].indicator_var))
        b, disj1, disj2, aux_vars1, aux_vars2 = self.check_transformation_block_structure(m, 0, 36, 0, 72, -9, 16, -18, 32)
        self.check_disjunct_constraints(disj1, disj2, aux_vars1, aux_vars2)
        c = b.component('disjunction_disjuncts[0].constraint[1]_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertIs(repn.linear_vars[0], aux_vars1[0])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])
        self.assertIsNone(repn.nonlinear_expr)
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], m.x[3], m.x[4])
        c = b.component('disjunction_disjuncts[1].constraint[1]_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertIs(repn.linear_vars[0], m.x[2])
        self.assertIs(repn.linear_vars[1], aux_vars2[0])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])
        self.assertIsNone(repn.nonlinear_expr)
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], m.x[3], m.x[4])

    @unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
    def test_transformation_block_arbitrary_odd_partition(self):
        m = models.makeBetweenStepsPaperExample()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, num_partitions=3, compute_bounds_solver=SolverFactory('gurobi_direct'), compute_bounds_method=compute_optimal_bounds)
        b, disj1, disj2 = self.check_transformation_block_disjuncts_and_constraints(m, m.disjunction)
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)
        aux_vars1 = disj1.component('disjunction_disjuncts[0].constraint[1]_aux_vars')
        self.assertEqual(len(aux_vars1), 3)
        self.assertEqual(aux_vars1[0].lb, 0)
        self.assertEqual(aux_vars1[0].ub, 72)
        self.assertEqual(aux_vars1[1].lb, 0)
        self.assertEqual(aux_vars1[1].ub, 36)
        self.assertEqual(aux_vars1[2].lb, 0)
        self.assertEqual(aux_vars1[2].ub, 36)
        aux_vars2 = disj2.component('disjunction_disjuncts[1].constraint[1]_aux_vars')
        self.assertEqual(len(aux_vars2), 3)
        self.assertEqual(aux_vars2[0].lb, -18)
        self.assertEqual(aux_vars2[0].ub, 32)
        self.assertEqual(aux_vars2[1].lb, -9)
        self.assertEqual(aux_vars2[1].ub, 16)
        self.assertEqual(aux_vars2[2].lb, -9)
        self.assertEqual(aux_vars2[2].ub, 16)
        c = disj1.component('disjunction_disjuncts[0].constraint[1]')
        self.assertEqual(len(c), 1)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(value(c1.upper), 1)
        repn = generate_standard_repn(c1.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        self.assertEqual(repn.constant, 0)
        self.assertIs(repn.linear_vars[0], aux_vars1[0])
        self.assertIs(repn.linear_vars[1], aux_vars1[1])
        self.assertIs(repn.linear_vars[2], aux_vars1[2])
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertEqual(repn.linear_coefs[1], 1)
        self.assertEqual(repn.linear_coefs[2], 1)
        c = disj2.component('disjunction_disjuncts[1].constraint[1]')
        self.assertEqual(len(c), 1)
        c2 = c[0]
        self.assertIsNone(c2.lower)
        self.assertEqual(value(c2.upper), -35)
        repn = generate_standard_repn(c2.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        self.assertEqual(repn.constant, 0)
        self.assertIs(repn.linear_vars[0], aux_vars2[0])
        self.assertIs(repn.linear_vars[1], aux_vars2[1])
        self.assertIs(repn.linear_vars[2], aux_vars2[2])
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertEqual(repn.linear_coefs[1], 1)
        self.assertEqual(repn.linear_coefs[2], 1)
        c = b.component('disjunction_disjuncts[0].constraint[1]_split_constraints')
        self.assertEqual(len(c), 3)
        c.pprint()
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], m.x[1], m.x[4])
        c2 = c[1]
        self.assertIsNone(c2.lower)
        self.assertEqual(c2.upper, 0)
        repn = generate_standard_repn(c2.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.linear_vars[0], aux_vars1[1])
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])
        c3 = c[2]
        self.assertIsNone(c3.lower)
        self.assertEqual(c3.upper, 0)
        repn = generate_standard_repn(c3.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.linear_vars[0], aux_vars1[2])
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[3])
        self.assertIs(repn.quadratic_vars[0][1], m.x[3])
        self.assertIsNone(repn.nonlinear_expr)
        c = b.component('disjunction_disjuncts[1].constraint[1]_split_constraints')
        self.assertEqual(len(c), 3)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], m.x[1], m.x[4])
        c2 = c[1]
        self.assertIsNone(c2.lower)
        self.assertEqual(c2.upper, 0)
        repn = generate_standard_repn(c2.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.linear_vars[0], m.x[2])
        self.assertIs(repn.linear_vars[1], aux_vars2[1])
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])
        c3 = c[2]
        self.assertIsNone(c3.lower)
        self.assertEqual(c3.upper, 0)
        repn = generate_standard_repn(c3.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.linear_vars[0], m.x[3])
        self.assertIs(repn.linear_vars[1], aux_vars2[2])
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[3])
        self.assertIs(repn.quadratic_vars[0][1], m.x[3])

    def test_transformed_disjuncts_mapped_correctly(self):
        m = models.makeBetweenStepsPaperExample()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_fbbt_bounds)
        b = m.component('_pyomo_gdp_partition_disjuncts_reformulation')
        self.assertIs(m.disjunction.disjuncts[0].transformation_block, b.disjunction.disjuncts[0])
        self.assertIs(m.disjunction.disjuncts[1].transformation_block, b.disjunction.disjuncts[1])

    def test_transformed_disjunctions_mapped_correctly(self):
        m = models.makeBetweenStepsPaperExample()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_fbbt_bounds)
        b = m.component('_pyomo_gdp_partition_disjuncts_reformulation')
        self.assertIs(m.disjunction.algebraic_constraint, b.disjunction)

    def add_disjunction(self, b):
        m = b.model()
        b.another_disjunction = Disjunction(expr=[[(m.x[1] - 1) ** 2 + m.x[2] ** 2 <= 1], [-(m.x[1] - 2) ** 2 - (m.x[2] - 3) ** 2 >= -1]])

    def make_model_with_added_disjunction_on_block(self):
        m = models.makeBetweenStepsPaperExample()
        m.b = Block()
        self.add_disjunction(m.b)
        return m

    def check_second_disjunction_aux_vars(self, aux_vars1, aux_vars2):
        self.assertEqual(len(aux_vars1), 2)
        self.assertEqual(aux_vars1[0].lb, -1)
        self.assertEqual(aux_vars1[0].ub, 24)
        self.assertEqual(aux_vars1[1].lb, 0)
        self.assertEqual(aux_vars1[1].ub, 36)
        self.assertEqual(len(aux_vars2), 2)
        self.assertEqual(aux_vars2[0].lb, -4)
        self.assertEqual(aux_vars2[0].ub, 12)
        self.assertEqual(aux_vars2[1].lb, -9)
        self.assertEqual(aux_vars2[1].ub, 16)

    def check_second_disjunction_global_constraint_disj1(self, c, aux_vars1):
        m = c.model()
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -2)
        self.assertIs(repn.linear_vars[0], m.x[1])
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertIs(repn.linear_vars[1], aux_vars1[0])
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[1])
        self.assertIs(repn.quadratic_vars[0][1], m.x[1])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIsNone(repn.nonlinear_expr)
        c2 = c[1]
        self.assertIsNone(c2.lower)
        self.assertEqual(c2.upper, 0)
        repn = generate_standard_repn(c2.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertIs(repn.linear_vars[0], aux_vars1[1])
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIsNone(repn.nonlinear_expr)

    def check_second_disjunction_global_constraint_disj2(self, c, aux_vars2):
        m = c.model()
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -4)
        self.assertIs(repn.linear_vars[0], m.x[1])
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertIs(repn.linear_vars[1], aux_vars2[0])
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[1])
        self.assertIs(repn.quadratic_vars[0][1], m.x[1])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIsNone(repn.nonlinear_expr)
        c2 = c[1]
        self.assertIsNone(c2.lower)
        self.assertEqual(c2.upper, 0)
        repn = generate_standard_repn(c2.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertIs(repn.linear_vars[0], m.x[2])
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertIs(repn.linear_vars[1], aux_vars2[1])
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIsNone(repn.nonlinear_expr)

    def test_disjunction_target(self):
        m = self.make_model_with_added_disjunction_on_block()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_fbbt_bounds, targets=[m.disjunction])
        self.check_transformation_block(m, 0, 72, 0, 72, -72, 96, -72, 96, partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]])
        self.assertIsNone(m.b.another_disjunction.algebraic_constraint)
        self.assertIsNone(m.b.another_disjunction.disjuncts[0].transformation_block)
        self.assertIsNone(m.b.another_disjunction.disjuncts[1].transformation_block)

    @unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
    def test_block_target(self):
        m = self.make_model_with_added_disjunction_on_block()
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1]], [m.x[2]]], compute_bounds_solver=SolverFactory('gurobi_direct'), compute_bounds_method=compute_optimal_bounds, targets=[m.b])
        self.assertIsNone(m.disjunction.algebraic_constraint)
        self.assertIsNone(m.disjunction.disjuncts[0].transformation_block)
        self.assertIsNone(m.disjunction.disjuncts[1].transformation_block)
        b = m.b.component('_pyomo_gdp_partition_disjuncts_reformulation')
        self.assertIsInstance(b, Block)
        self.assertEqual(len(b.component_map(Disjunction)), 1)
        self.assertEqual(len(b.component_map(Disjunct)), 2)
        self.assertEqual(len(b.component_map(Constraint)), 2)
        disjunction = b.component('b.another_disjunction')
        self.assertIsInstance(disjunction, Disjunction)
        self.assertEqual(len(disjunction.disjuncts), 2)
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)
        aux_vars1 = disj1.component('b.another_disjunction_disjuncts[0].constraint[1]_aux_vars')
        aux_vars2 = disj2.component('b.another_disjunction_disjuncts[1].constraint[1]_aux_vars')
        self.check_second_disjunction_aux_vars(aux_vars1, aux_vars2)
        c1 = disj1.component('b.another_disjunction_disjuncts[0].constraint[1]')
        self.assertEqual(len(c1), 1)
        self.check_disj_constraint(c1[0], 0, aux_vars1[0], aux_vars1[1])
        c2 = disj2.component('b.another_disjunction_disjuncts[1].constraint[1]')
        self.assertEqual(len(c2), 1)
        self.check_disj_constraint(c2[0], -12, aux_vars2[0], aux_vars2[1])
        c = b.component('b.another_disjunction_disjuncts[0].constraint[1]_split_constraints')
        self.check_second_disjunction_global_constraint_disj1(c, aux_vars1)
        c = b.component('b.another_disjunction_disjuncts[1].constraint[1]_split_constraints')
        self.check_second_disjunction_global_constraint_disj2(c, aux_vars2)

    @unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
    def test_indexed_block_target(self):
        m = ConcreteModel()
        m.b = Block(Any)
        m.b[0].transfer_attributes_from(models.makeBetweenStepsPaperExample())
        m.x = Reference(m.b[0].x)
        self.add_disjunction(m.b[1])
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions={m.b[1].another_disjunction: [[m.x[1]], [m.x[2]]], m.b[0].disjunction: [[m.x[1], m.x[2]], [m.x[3], m.x[4]]]}, compute_bounds_solver=SolverFactory('gurobi_direct'), compute_bounds_method=compute_optimal_bounds, targets=[m.b])
        b0 = m.b[0].component('_pyomo_gdp_partition_disjuncts_reformulation')
        self.assertIsInstance(b0, Block)
        self.assertEqual(len(b0.component_map(Disjunction)), 1)
        self.assertEqual(len(b0.component_map(Disjunct)), 2)
        self.assertEqual(len(b0.component_map(Constraint)), 2)
        b1 = m.b[1].component('_pyomo_gdp_partition_disjuncts_reformulation')
        self.assertIsInstance(b1, Block)
        self.assertEqual(len(b1.component_map(Disjunction)), 1)
        self.assertEqual(len(b1.component_map(Disjunct)), 2)
        self.assertEqual(len(b1.component_map(Constraint)), 2)
        disjunction = b1.component('b[1].another_disjunction')
        self.assertIsInstance(disjunction, Disjunction)
        self.assertEqual(len(disjunction.disjuncts), 2)
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)
        aux_vars1 = disj1.component('b[1].another_disjunction_disjuncts[0].constraint[1]_aux_vars')
        aux_vars2 = disj2.component('b[1].another_disjunction_disjuncts[1].constraint[1]_aux_vars')
        self.check_second_disjunction_aux_vars(aux_vars1, aux_vars2)
        c1 = disj1.component('b[1].another_disjunction_disjuncts[0].constraint[1]')
        self.assertEqual(len(c1), 1)
        self.check_disj_constraint(c1[0], 0, aux_vars1[0], aux_vars1[1])
        c2 = disj2.component('b[1].another_disjunction_disjuncts[1].constraint[1]')
        self.assertEqual(len(c2), 1)
        self.check_disj_constraint(c2[0], -12, aux_vars2[0], aux_vars2[1])
        c = b1.component('b[1].another_disjunction_disjuncts[0].constraint[1]_split_constraints')
        self.check_second_disjunction_global_constraint_disj1(c, aux_vars1)
        c = b1.component('b[1].another_disjunction_disjuncts[1].constraint[1]_split_constraints')
        self.check_second_disjunction_global_constraint_disj2(c, aux_vars2)
        disjunction = b0.component('b[0].disjunction')
        self.assertIsInstance(disjunction, Disjunction)
        self.assertEqual(len(disjunction.disjuncts), 2)
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)
        aux_vars1 = disj1.component('b[0].disjunction_disjuncts[0].constraint[1]_aux_vars')
        aux_vars2 = disj2.component('b[0].disjunction_disjuncts[1].constraint[1]_aux_vars')
        self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -18, 32, -18, 32)
        c1 = disj1.component('b[0].disjunction_disjuncts[0].constraint[1]')
        self.assertEqual(len(c1), 1)
        self.check_disj_constraint(c1[0], 1, aux_vars1[0], aux_vars1[1])
        c2 = disj2.component('b[0].disjunction_disjuncts[1].constraint[1]')
        self.assertEqual(len(c2), 1)
        self.check_disj_constraint(c2[0], -35, aux_vars2[0], aux_vars2[1])
        c = b0.component('b[0].disjunction_disjuncts[0].constraint[1]_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], m.x[3], m.x[4])
        c = b0.component('b[0].disjunction_disjuncts[1].constraint[1]_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], m.x[3], m.x[4])

    @unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
    def test_indexed_disjunction_target(self):
        m = ConcreteModel()
        m.I = RangeSet(1, 4)
        m.x = Var(m.I, bounds=(-2, 6))
        m.indexed = Disjunction(Any)
        m.indexed[1] = [[sum((m.x[i] ** 2 for i in m.I)) <= 1], [sum(((3 - m.x[i]) ** 2 for i in m.I)) <= 1]]
        m.indexed[0] = [[(m.x[1] - 1) ** 2 + m.x[2] ** 2 <= 1], [-(m.x[1] - 2) ** 2 - (m.x[2] - 3) ** 2 >= -1]]
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions={m.indexed[0]: [[m.x[1]], [m.x[2]]], m.indexed[1]: [[m.x[1], m.x[2]], [m.x[3], m.x[4]]]}, compute_bounds_solver=SolverFactory('gurobi_direct'), compute_bounds_method=compute_optimal_bounds, targets=[m.indexed])
        b = m.component('_pyomo_gdp_partition_disjuncts_reformulation')
        self.assertIsInstance(b, Block)
        self.assertEqual(len(b.component_map(Disjunction)), 2)
        self.assertEqual(len(b.component_map(Disjunct)), 4)
        self.assertEqual(len(b.component_map(Constraint)), 4)
        disjunction = b.component('indexed[0]')
        self.assertIsInstance(disjunction, Disjunction)
        self.assertEqual(len(disjunction.disjuncts), 2)
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)
        aux_vars1 = disj1.component('indexed_disjuncts[2].constraint[1]_aux_vars')
        aux_vars2 = disj2.component('indexed_disjuncts[3].constraint[1]_aux_vars')
        self.check_second_disjunction_aux_vars(aux_vars1, aux_vars2)
        c1 = disj1.component('indexed_disjuncts[2].constraint[1]')
        self.assertEqual(len(c1), 1)
        self.check_disj_constraint(c1[0], 0, aux_vars1[0], aux_vars1[1])
        c2 = disj2.component('indexed_disjuncts[3].constraint[1]')
        self.assertEqual(len(c2), 1)
        self.check_disj_constraint(c2[0], -12, aux_vars2[0], aux_vars2[1])
        c = b.component('indexed_disjuncts[2].constraint[1]_split_constraints')
        self.check_second_disjunction_global_constraint_disj1(c, aux_vars1)
        c = b.component('indexed_disjuncts[3].constraint[1]_split_constraints')
        self.check_second_disjunction_global_constraint_disj2(c, aux_vars2)
        disjunction = b.component('indexed[1]')
        self.assertIsInstance(disjunction, Disjunction)
        self.assertEqual(len(disjunction.disjuncts), 2)
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)
        aux_vars1 = disj1.component('indexed_disjuncts[0].constraint[1]_aux_vars')
        aux_vars2 = disj2.component('indexed_disjuncts[1].constraint[1]_aux_vars')
        self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -18, 32, -18, 32)
        c1 = disj1.component('indexed_disjuncts[0].constraint[1]')
        self.assertEqual(len(c1), 1)
        self.check_disj_constraint(c1[0], 1, aux_vars1[0], aux_vars1[1])
        c2 = disj2.component('indexed_disjuncts[1].constraint[1]')
        self.assertEqual(len(c2), 1)
        self.check_disj_constraint(c2[0], -35, aux_vars2[0], aux_vars2[1])
        c = b.component('indexed_disjuncts[0].constraint[1]_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], m.x[3], m.x[4])
        c = b.component('indexed_disjuncts[1].constraint[1]_split_constraints')
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], m.x[3], m.x[4])

    def test_incomplete_partition_error(self):
        m = models.makeBetweenStepsPaperExample()
        self.assertRaisesRegex(GDP_Error, "Partition specified for disjunction containing Disjunct 'disjunction_disjuncts\\[0\\]' does not include all the variables that appear in the disjunction. The following variables are not assigned to any part of the partition: 'x\\[3\\]', 'x\\[4\\]'", TransformationFactory('gdp.partition_disjuncts').apply_to, m, variable_partitions=[[m.x[1]], [m.x[2]]], compute_bounds_method=compute_fbbt_bounds)

    def test_unbounded_expression_error(self):
        m = models.makeBetweenStepsPaperExample()
        for i in m.x:
            m.x[i].setub(None)
        self.assertRaisesRegex(GDP_Error, "Expression x\\[1\\]\\*x\\[1\\] from constraint 'disjunction_disjuncts\\[0\\].constraint\\[1\\]' is unbounded! Please ensure all variables that appear in the constraint are bounded or specify compute_bounds_method=compute_optimal_bounds if the expression is bounded by the global constraints.", TransformationFactory('gdp.partition_disjuncts').apply_to, m, variable_partitions=[[m.x[1]], [m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_fbbt_bounds)

    def test_no_value_for_P_error(self):
        m = models.makeBetweenStepsPaperExample()
        with self.assertRaisesRegex(GDP_Error, 'No value for P was given for disjunction disjunction! Please specify a value of P \\(number of partitions\\), if you do not specify the partitions directly.'):
            TransformationFactory('gdp.partition_disjuncts').apply_to(m)

    def test_create_using(self):
        m = models.makeBetweenStepsPaperExample()
        self.diff_apply_to_and_create_using(m, num_partitions=2)