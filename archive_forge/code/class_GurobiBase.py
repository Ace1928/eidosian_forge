import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
class GurobiBase(unittest.TestCase):

    def setUp(self):
        clean_up_global_state()
        model = ConcreteModel()
        model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
        model.OBJ = pyo.Objective(expr=model.x[1] + model.x[2], sense=pyo.maximize)
        model.Constraint1 = pyo.Constraint(expr=2 * model.x[1] + model.x[2] <= 1)
        model.Constraint2 = pyo.Constraint(expr=model.x[1] + 2 * model.x[2] <= 1)
        self.model = model

    def tearDown(self):
        clean_up_global_state()