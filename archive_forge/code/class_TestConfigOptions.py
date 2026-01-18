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
@unittest.skipIf(not LOA_solvers_available, 'Required subsolvers %s are not available' % (LOA_solvers,))
class TestConfigOptions(unittest.TestCase):

    def make_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-10, 10))
        m.y = Var(bounds=(-10, 10))
        m.disjunction = Disjunction(expr=[[m.y >= m.x ** 2 + 1], [m.y >= m.x ** 2 - 3 * m.x + 2]])
        m.obj = Objective(expr=m.y)
        return m

    @unittest.skipIf(not mcpp_available(), 'MC++ is not available')
    @unittest.skipUnless(gurobi_available, 'Gurobi not available')
    def test_set_options_on_config_block(self):
        m = self.make_model()
        opt = SolverFactory('gdpopt.loa')
        with self.assertRaisesRegex(ValueError, "Changing the algorithm in the solve method is not supported for algorithm-specific GDPopt solvers. Either use SolverFactory[(]'gdpopt'[)] or instantiate a solver with the algorithm you want to use."):
            opt.solve(m, algorithm='RIC')
        opt.CONFIG.mip_solver = mip_solver
        opt.CONFIG.nlp_solver = nlp_solver
        opt.CONFIG.init_algorithm = 'no_init'
        buf = StringIO()
        with redirect_stdout(buf):
            opt.solve(m, tee=True)
        self.assertIn('mip_solver: gurobi', buf.getvalue())
        self.assertAlmostEqual(value(m.obj), -0.25)
        opt = SolverFactory('gdpopt.loa')
        opt.config.mip_solver = 'glpk'
        self.assertEqual(opt.CONFIG.mip_solver, mip_solver)
        self.assertEqual(opt.CONFIG.nlp_solver, nlp_solver)
        self.assertEqual(opt.CONFIG.init_algorithm, 'no_init')
        buf = StringIO()
        with redirect_stdout(buf):
            opt.solve(m, tee=True)
        self.assertIn('mip_solver: glpk', buf.getvalue())
        self.assertAlmostEqual(value(m.obj), -0.25)

    @unittest.skipIf(not mcpp_available(), 'MC++ is not available')
    @unittest.skipUnless(gurobi_available, 'Gurobi not available')
    def test_set_options_in_init(self):
        m = self.make_model()
        opt = SolverFactory('gdpopt.loa', mip_solver='gurobi', nlp_solver=nlp_solver, init_algorithm='no_init')
        buf = StringIO()
        with redirect_stdout(buf):
            opt.solve(m, tee=True)
        self.assertIn('mip_solver: gurobi', buf.getvalue())
        self.assertAlmostEqual(value(m.obj), -0.25)
        buf = StringIO()
        with redirect_stdout(buf):
            opt.solve(m, tee=True, mip_solver='glpk')
        self.assertIn('mip_solver: glpk', buf.getvalue())
        self.assertAlmostEqual(value(m.obj), -0.25)
        self.assertEqual(opt.config.mip_solver, 'gurobi')

    @unittest.skipUnless(gurobi_available, 'Gurobi not available')
    def test_no_default_algorithm(self):
        m = self.make_model()
        opt = SolverFactory('gdpopt')
        buf = StringIO()
        with redirect_stdout(buf):
            opt.solve(m, algorithm='RIC', tee=True, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertIn('using RIC algorithm', buf.getvalue())
        self.assertAlmostEqual(value(m.obj), -0.25)
        buf = StringIO()
        with redirect_stdout(buf):
            opt.solve(m, algorithm='LBB', tee=True, mip_solver=mip_solver, nlp_solver=nlp_solver, minlp_solver='gurobi')
        self.assertIn('using LBB algorithm', buf.getvalue())
        self.assertAlmostEqual(value(m.obj), -0.25)