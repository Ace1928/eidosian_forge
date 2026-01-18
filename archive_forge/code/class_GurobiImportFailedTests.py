import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
@unittest.skipIf(gurobipy_available, 'gurobipy is installed, skip import test')
class GurobiImportFailedTests(unittest.TestCase):

    def test_gurobipy_not_installed(self):
        model = ConcreteModel()
        with SolverFactory('gurobi_direct') as opt:
            with self.assertRaisesRegex(ApplicationError, 'No Python bindings'):
                opt.solve(model)