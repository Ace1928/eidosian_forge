from contextlib import redirect_stdout
from io import StringIO
import logging
from math import fabs
from os.path import join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.contrib.appsi.solvers.gurobi import Gurobi
from pyomo.contrib.gdpopt.create_oa_subproblems import (
import pyomo.contrib.gdpopt.tests.common_tests as ct
from pyomo.contrib.gdpopt.util import is_feasible, time_code
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.tests import models
from pyomo.opt import TerminationCondition
@unittest.skipIf(not LOA_solvers_available, 'Required subsolvers %s are not available' % (LOA_solvers,))
class TestGDPoptRIC(unittest.TestCase):
    """Tests for the GDPopt solver plugin."""

    def test_infeasible_GDP(self):
        """Test for infeasible GDP."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 2))
        m.d = Disjunction(expr=[[m.x ** 2 >= 3, m.x >= 3], [m.x ** 2 <= -1, m.x <= -1]])
        m.o = Objective(expr=m.x)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            SolverFactory('gdpopt.ric').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
            self.assertIn('Set covering problem is infeasible.', output.getvalue().strip())

    def test_GDP_nonlinear_objective(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.d = Disjunction(expr=[[m.x + m.y >= 5], [m.x - m.y <= 3]])
        m.o = Objective(expr=m.x ** 2)
        SolverFactory('gdpopt.ric').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.o), 0)
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.d = Disjunction(expr=[[m.x + m.y >= 5], [m.x - m.y <= 3]])
        m.o = Objective(expr=-m.x ** 2, sense=maximize)
        SolverFactory('gdpopt.ric').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.o), 0)

    def test_logical_constraints_on_disjuncts(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        SolverFactory('gdpopt.ric').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    def test_boolean_vars_on_disjuncts(self):
        m = models.makeBooleanVarsOnDisjuncts()
        SolverFactory('gdpopt.ric').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    def test_RIC_8PP_default_init(self):
        """Test logic-based outer approximation with 8PP."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.ric').solve(eight_process, mip_solver=mip_solver, nlp_solver=nlp_solver, tee=False)
        ct.check_8PP_solution(self, eight_process, results)

    def test_RIC_8PP_logical_default_init(self):
        """Test logic-based outer approximation with 8PP."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_logical.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.ric').solve(eight_process, mip_solver=mip_solver, nlp_solver=nlp_solver, tee=False)
        ct.check_8PP_logical_solution(self, eight_process, results)

    @unittest.skipUnless(SolverFactory('gams').available(exception_flag=False), 'GAMS solver not available')
    def test_RIC_8PP_gams_solver(self):
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.ric').solve(eight_process, mip_solver=mip_solver, nlp_solver='gams', max_slack=0, tee=False)
        ct.check_8PP_solution(self, eight_process, results)

    def test_RIC_8PP_force_NLP(self):
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.ric').solve(eight_process, mip_solver=mip_solver, nlp_solver=nlp_solver, force_subproblem_nlp=True, tee=False)
        ct.check_8PP_solution(self, eight_process, results)

    def test_RIC_strip_pack_default_init(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt.ric').solve(strip_pack, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 11) <= 0.01)

    def test_RIC_strip_pack_default_init_logical_constraints(self):
        """Test logic-based outer approximation with strip packing with
        logical constraints."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        strip_pack.Rec3AboveOrBelowRec1 = LogicalConstraint(expr=strip_pack.no_overlap[1, 3].disjuncts[2].indicator_var.lor(strip_pack.no_overlap[1, 3].disjuncts[3].indicator_var))
        strip_pack.Rec3RightOrLeftOfRec2 = LogicalConstraint(expr=strip_pack.no_overlap[2, 3].disjuncts[0].indicator_var.lor(strip_pack.no_overlap[2, 3].disjuncts[1].indicator_var))
        SolverFactory('gdpopt.ric').solve(strip_pack, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 13) <= 0.01)

    @unittest.pytest.mark.expensive
    def test_RIC_constrained_layout_default_init(self):
        """Test RIC with constrained layout."""
        exfile = import_file(join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt.ric').solve(cons_layout, mip_solver=mip_solver, nlp_solver=nlp_solver, iterlim=120, max_slack=5)
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(fabs(objective_value - 41573) <= 200, 'Objective value of %s instead of 41573' % objective_value)

    def test_RIC_8PP_maxBinary(self):
        """Test logic-based OA with max_binary initialization."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.ric').solve(eight_process, init_algorithm='max_binary', mip_solver=mip_solver, nlp_solver=nlp_solver)
        ct.check_8PP_solution(self, eight_process, results)

    def test_RIC_strip_pack_maxBinary(self):
        """Test RIC with strip packing using max_binary initialization."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt.ric').solve(strip_pack, init_algorithm='max_binary', mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 11) <= 0.01)

    def test_RIC_strip_pack_maxBinary_logical_constraints(self):
        """Test RIC with strip packing using max_binary initialization and
        including logical constraints."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        strip_pack.Rec3AboveOrBelowRec1 = LogicalConstraint(expr=strip_pack.no_overlap[1, 3].disjuncts[2].indicator_var.lor(strip_pack.no_overlap[1, 3].disjuncts[3].indicator_var))
        strip_pack.Rec3RightOrLeftOfRec2 = LogicalConstraint(expr=strip_pack.no_overlap[2, 3].disjuncts[0].indicator_var.lor(strip_pack.no_overlap[2, 3].disjuncts[1].indicator_var))
        SolverFactory('gdpopt.ric').solve(strip_pack, init_algorithm='max_binary', mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 13) <= 0.01)

    def test_RIC_8PP_fixed_disjuncts(self):
        """Test RIC with 8PP using fixed disjuncts initialization."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        initialize = [eight_process.use_unit_1or2.disjuncts[0], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[1], eight_process.use_unit_8ornot.disjuncts[0]]
        for disj in eight_process.component_data_objects(Disjunct):
            if disj in initialize:
                disj.binary_indicator_var.set_value(1)
            else:
                disj.binary_indicator_var.set_value(0)
        results = SolverFactory('gdpopt.ric').solve(eight_process, init_algorithm='fix_disjuncts', mip_solver=mip_solver, nlp_solver=nlp_solver)
        ct.check_8PP_solution(self, eight_process, results)

    def test_RIC_custom_disjuncts(self):
        """Test logic-based OA with custom disjuncts initialization."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        initialize = [[eight_process.use_unit_1or2.disjuncts[0], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[1], eight_process.use_unit_8ornot.disjuncts[0]], [eight_process.use_unit_1or2.disjuncts[1], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[0], eight_process.use_unit_8ornot.disjuncts[0]]]

        def assert_correct_disjuncts_active(solver, subprob_util_block, discrete_problem_util_block):
            iteration = solver.initialization_iteration
            discrete_problem = discrete_problem_util_block.model()
            subprob = subprob_util_block.model()
            if iteration >= 2:
                return
            disjs_should_be_active = initialize[iteration]
            seen = set()
            for orig_disj in disjs_should_be_active:
                parent_nm = orig_disj.parent_component().name
                idx = orig_disj.index()
                discrete_problem_parent = discrete_problem.component(parent_nm)
                subprob_parent = subprob.component(parent_nm)
                self.assertIsInstance(discrete_problem_parent, Disjunct)
                self.assertIsInstance(subprob_parent, Block)
                discrete_problem_disj = discrete_problem_parent[idx]
                subprob_disj = subprob_parent[idx]
                self.assertTrue(value(discrete_problem_disj.indicator_var))
                self.assertTrue(subprob_disj.active)
                seen.add(subprob_disj)
            for disj in subprob_util_block.disjunct_list:
                if disj not in seen:
                    self.assertFalse(disj.active)
        SolverFactory('gdpopt.ric').solve(eight_process, init_algorithm='custom_disjuncts', custom_init_disjuncts=initialize, mip_solver=mip_solver, nlp_solver=nlp_solver, subproblem_initialization_method=assert_correct_disjuncts_active)
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 0.01)

    def test_force_nlp_subproblem_with_general_integer_variables(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers, bounds=(0, 10))
        m.y = Var(bounds=(0, 10))
        m.disjunction = Disjunction(expr=[[m.x ** 2 <= 4, m.y ** 2 <= 1], [(m.x - 1) ** 2 + (m.y - 1) ** 2 <= 4, m.y <= 4]])
        m.obj = Objective(expr=-m.y - m.x)
        results = SolverFactory('gdpopt.ric').solve(m, init_algorithm='no_init', mip_solver=mip_solver, nlp_solver=nlp_solver, force_subproblem_nlp=True)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.x), 2)
        self.assertAlmostEqual(value(m.y), 1 + sqrt(3))

    def test_force_nlp_subproblem_with_unbounded_integer_variables(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers, bounds=(0, 10))
        m.y = Var(bounds=(0, 10))
        m.w = Var(domain=Integers)
        m.disjunction = Disjunction(expr=[[m.x ** 2 <= 4, m.y ** 2 <= 1], [(m.x - 1) ** 2 + (m.y - 1) ** 2 <= 4, m.y <= 4]])
        m.c = Constraint(expr=m.x + m.y == m.w)
        m.obj = Objective(expr=-m.w)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            results = SolverFactory('gdpopt.ric').solve(m, init_algorithm='no_init', mip_solver=mip_solver, nlp_solver=nlp_solver, force_subproblem_nlp=True, iterlim=5)
        self.assertIn('No feasible solutions found.', output.getvalue().strip())
        self.assertEqual(results.solver.termination_condition, TerminationCondition.maxIterations)
        self.assertIsNone(m.x.value)
        self.assertIsNone(m.y.value)
        self.assertIsNone(m.w.value)
        self.assertIsNone(m.disjunction.disjuncts[0].indicator_var.value)
        self.assertIsNone(m.disjunction.disjuncts[1].indicator_var.value)
        self.assertIsNotNone(results.solver.user_time)