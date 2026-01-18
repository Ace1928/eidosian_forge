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
class GurobiParameterTests(GurobiBase):

    def test_set_environment_parameters(self):
        with SolverFactory('gurobi_direct', manage_env=True, options={'ComputeServer': 'my-cs-url'}) as opt:
            with self.assertRaisesRegex(ApplicationError, 'my-cs-url'):
                opt.solve(self.model)

    def test_set_once(self):
        envparams = {}
        modelparams = {}

        class TempEnv(gp.Env):

            def setParam(self, param, value):
                envparams[param] = value

        class TempModel(gp.Model):

            def setParam(self, param, value):
                modelparams[param] = value
        with patch('gurobipy.Env', new=TempEnv), patch('gurobipy.Model', new=TempModel):
            with SolverFactory('gurobi_direct', options={'Method': 2, 'MIPFocus': 1}, manage_env=True) as opt:
                opt.solve(self.model, options={'MIPFocus': 2})
        assert envparams == {'Method': 2, 'MIPFocus': 1}
        assert modelparams == {'MIPFocus': 2, 'OutputFlag': 0}

    def test_param_changes_1(self):
        with SolverFactory('gurobi_direct', options={'Method': -100}) as opt:
            with self.assertRaisesRegex(gp.GurobiError, 'Unable to set'):
                opt.solve(self.model)

    def test_param_changes_2(self):
        with SolverFactory('gurobi_direct', options={'Method': -100}, manage_env=True) as opt:
            with self.assertRaisesRegex(ApplicationError, 'Unable to set'):
                opt.solve(self.model)

    def test_param_changes_3(self):
        with SolverFactory('gurobi_direct') as opt:
            with self.assertRaisesRegex(gp.GurobiError, 'Unable to set'):
                opt.solve(self.model, options={'Method': -100})

    def test_param_changes_4(self):
        with SolverFactory('gurobi_direct', manage_env=True) as opt:
            with self.assertRaisesRegex(gp.GurobiError, 'Unable to set'):
                opt.solve(self.model, options={'Method': -100})