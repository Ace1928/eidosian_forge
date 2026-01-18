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
@unittest.skipIf(not GLOA_solvers_available, 'Required subsolvers %s are not available' % (GLOA_solvers,))
@unittest.skipIf(not mcpp_available(), 'MC++ is not available')
class TestGLOA(unittest.TestCase):
    """Tests for global logic-based outer approximation."""

    def test_GDP_integer_vars(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(domain=Integers, bounds=(0, 5))
        m.d = Disjunction(expr=[[m.x >= m.y, m.y >= 3.5], [m.x >= m.y, m.y >= 2.5]])
        m.o = Objective(expr=m.x)
        SolverFactory('gdpopt').solve(m, algorithm='GLOA', mip_solver=mip_solver, nlp_solver=global_nlp_solver, minlp_solver=minlp_solver)
        self.assertAlmostEqual(value(m.o.expr), 3)

    def make_nonlinear_gdp_with_int_vars(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(domain=Integers, bounds=(0, 5))
        m.d = Disjunction(expr=[[m.x ** 2 >= m.y, m.y >= 3.5], [m.x ** 2 >= m.y, m.y >= 2.5]])
        m.o = Objective(expr=m.x)
        return m

    def test_nonlinear_GDP_integer_vars(self):
        m = self.make_nonlinear_gdp_with_int_vars()
        SolverFactory('gdpopt.gloa').solve(m, mip_solver=mip_solver, nlp_solver=global_nlp_solver, minlp_solver=minlp_solver)
        self.assertAlmostEqual(value(m.o.expr), sqrt(3))
        self.assertAlmostEqual(value(m.y), 3)

    def test_nonlinear_GDP_integer_vars_force_nlp_subproblem(self):
        m = self.make_nonlinear_gdp_with_int_vars()
        SolverFactory('gdpopt.gloa').solve(m, mip_solver=mip_solver, nlp_solver=global_nlp_solver, minlp_solver=minlp_solver, force_subproblem_nlp=True)
        self.assertAlmostEqual(value(m.o.expr), sqrt(3))
        self.assertAlmostEqual(value(m.y), 3)

    def test_GDP_integer_vars_infeasible(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.y = Var(domain=Integers, bounds=(0, 5))
        m.d = Disjunction(expr=[[m.x >= m.y, m.y >= 3.5], [m.x >= m.y, m.y >= 2.5]])
        m.o = Objective(expr=m.x)
        res = SolverFactory('gdpopt.gloa').solve(m, mip_solver=mip_solver, nlp_solver=global_nlp_solver, minlp_solver=minlp_solver)
        self.assertEqual(res.solver.termination_condition, TerminationCondition.infeasible)

    @unittest.skipUnless(license_available, 'Global NLP solver license not available')
    def test_logical_constraints_on_disjuncts(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        SolverFactory('gdpopt.gloa').solve(m, mip_solver=mip_solver, nlp_solver=global_nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    @unittest.skipUnless(license_available, 'Global NLP solver license not available')
    def test_boolean_vars_on_disjuncts(self):
        m = models.makeBooleanVarsOnDisjuncts()
        SolverFactory('gdpopt.gloa').solve(m, mip_solver=mip_solver, nlp_solver=global_nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    @unittest.skipUnless(license_available, 'Global NLP solver license not available.')
    def test_GLOA_8PP(self):
        """Test the global logic-based outer approximation algorithm."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.gloa').solve(eight_process, tee=False, mip_solver=mip_solver, nlp_solver=global_nlp_solver, nlp_solver_args=global_nlp_solver_args)
        ct.check_8PP_solution(self, eight_process, results)

    @unittest.skipUnless(license_available, 'Global NLP solver license not available')
    def test_GLOA_8PP_logical(self):
        """Test the global logic-based outer approximation algorithm."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_logical.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.gloa').solve(eight_process, tee=False, mip_solver=mip_solver, nlp_solver=global_nlp_solver, nlp_solver_args=global_nlp_solver_args)
        ct.check_8PP_logical_solution(self, eight_process, results)

    @unittest.skipUnless(license_available, 'Global NLP solver license not available.')
    def test_GLOA_8PP_force_NLP(self):
        """Test the global logic-based outer approximation algorithm."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.gloa').solve(eight_process, tee=False, mip_solver=mip_solver, nlp_solver=global_nlp_solver, nlp_solver_args=global_nlp_solver_args, force_subproblem_nlp=True)
        ct.check_8PP_solution(self, eight_process, results)

    @unittest.skipUnless(license_available, 'Global NLP solver license not available.')
    def test_GLOA_strip_pack_default_init(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt.gloa').solve(strip_pack, mip_solver=mip_solver, nlp_solver=global_nlp_solver, nlp_solver_args=global_nlp_solver_args)
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 11) <= 0.01)

    @unittest.skipUnless(license_available, 'Global NLP solver license not available')
    def test_GLOA_strip_pack_default_init_logical_constraints(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        strip_pack.Rec3AboveOrBelowRec1 = LogicalConstraint(expr=strip_pack.no_overlap[1, 3].disjuncts[2].indicator_var.lor(strip_pack.no_overlap[1, 3].disjuncts[3].indicator_var))
        strip_pack.Rec3RightOrLeftOfRec2 = LogicalConstraint(expr=strip_pack.no_overlap[2, 3].disjuncts[0].indicator_var.lor(strip_pack.no_overlap[2, 3].disjuncts[1].indicator_var))
        SolverFactory('gdpopt.gloa').solve(strip_pack, mip_solver=mip_solver, nlp_solver=global_nlp_solver, nlp_solver_args=global_nlp_solver_args)
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 13) <= 0.01)

    @unittest.skipUnless(license_available, 'Global NLP solver license not available.')
    @unittest.pytest.mark.expensive
    def test_GLOA_constrained_layout_default_init(self):
        """Test LOA with constrained layout."""
        exfile = import_file(join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        results = SolverFactory('gdpopt.gloa').solve(cons_layout, mip_solver=mip_solver, nlp_solver=global_nlp_solver, nlp_solver_args=global_nlp_solver_args, tee=False)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(fabs(objective_value - 41573) <= 200, 'Objective value of %s instead of 41573' % objective_value)

    def test_GLOA_ex_633_trespalacios(self):
        """Test LOA with Francisco thesis example."""
        exfile = import_file(join(exdir, 'small_lit', 'ex_633_trespalacios.py'))
        model = exfile.build_simple_nonconvex_gdp()
        SolverFactory('gdpopt.gloa').solve(model, mip_solver=mip_solver, nlp_solver=global_nlp_solver, nlp_solver_args=global_nlp_solver_args, tee=False)
        objective_value = value(model.obj.expr)
        self.assertAlmostEqual(objective_value, 4.46, 2)

    @unittest.skipUnless(license_available, 'Global NLP solver license not available.')
    def test_GLOA_nonconvex_HENS(self):
        exfile = import_file(join(exdir, 'small_lit', 'nonconvex_HEN.py'))
        model = exfile.build_gdp_model()
        SolverFactory('gdpopt.gloa').solve(model, mip_solver=mip_solver, nlp_solver=global_nlp_solver, nlp_solver_args=global_nlp_solver_args, tee=False)
        objective_value = value(model.objective.expr)
        self.assertAlmostEqual(objective_value * 1e-05, 1.14385, 2)

    @unittest.skipUnless(license_available, 'Global NLP solver license not available.')
    def test_GLOA_disjunctive_bounds(self):
        exfile = import_file(join(exdir, 'small_lit', 'nonconvex_HEN.py'))
        model = exfile.build_gdp_model()
        SolverFactory('gdpopt.gloa').solve(model, mip_solver=mip_solver, nlp_solver=global_nlp_solver, nlp_solver_args=global_nlp_solver_args, calc_disjunctive_bounds=True, tee=False)
        objective_value = value(model.objective.expr)
        self.assertAlmostEqual(objective_value * 1e-05, 1.14385, 2)