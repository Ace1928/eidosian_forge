import pyomo.environ as pyo
from pyomo.common.fileutils import ExecutableData
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver.ipopt import IpoptConfig
from pyomo.contrib.solver.factory import SolverFactory
from pyomo.common import unittest
class TestIpopt(unittest.TestCase):

    def create_model(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(initialize=1.5)
        model.y = pyo.Var(initialize=1.5)

        def rosenbrock(m):
            return (1.0 - m.x) ** 2 + 100.0 * (m.y - m.x ** 2) ** 2
        model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)
        return model

    def test_ipopt_config(self):
        config = IpoptConfig()
        self.assertTrue(config.load_solutions)
        self.assertIsInstance(config.solver_options, ConfigDict)
        self.assertIsInstance(config.executable, ExecutableData)
        solver = SolverFactory('ipopt_v2', executable='/path/to/exe')
        self.assertFalse(solver.config.tee)
        self.assertTrue(solver.config.executable.startswith('/path'))