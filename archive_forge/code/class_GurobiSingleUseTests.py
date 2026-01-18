import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
@unittest.skipIf(not gurobipy_available, 'gurobipy is not available')
@unittest.skipIf(not gurobi_available, 'gurobi license is not valid')
@unittest.skipIf(not single_use_license(), reason='test needs a single use license')
class GurobiSingleUseTests(GurobiBase):

    def test_persisted_license_failure(self):
        with SolverFactory('gurobi_direct') as opt:
            with gp.Env():
                with self.assertRaises(ApplicationError):
                    opt.solve(self.model)
            opt.solve(self.model)

    def test_persisted_license_failure_managed(self):
        with SolverFactory('gurobi_direct', manage_env=True) as opt:
            with gp.Env():
                with self.assertRaises(ApplicationError):
                    opt.solve(self.model)
            opt.solve(self.model)

    def test_context(self):
        with SolverFactory('gurobi_direct', manage_env=True) as opt:
            opt.solve(self.model)
        with gp.Env():
            pass

    def test_close(self):
        opt = SolverFactory('gurobi_direct', manage_env=True)
        try:
            opt.solve(self.model)
        finally:
            opt.close()
        with gp.Env():
            pass

    def test_multiple_solvers(self):
        with SolverFactory('gurobi_direct') as opt1, SolverFactory('gurobi_direct') as opt2:
            opt1.solve(self.model)
            opt2.solve(self.model)

    def test_multiple_models_leaky(self):
        with SolverFactory('gurobi_direct', manage_env=True) as opt:
            opt.solve(self.model)
            tmp = opt._solver_model
            opt.solve(self.model)
        with gp.Env():
            pass

    def test_close_global(self):
        opt1 = SolverFactory('gurobi_direct')
        opt2 = SolverFactory('gurobi_direct')
        try:
            opt1.solve(self.model)
            opt2.solve(self.model)
        finally:
            opt1.close()
            opt2.close_global()
        with gp.Env():
            pass