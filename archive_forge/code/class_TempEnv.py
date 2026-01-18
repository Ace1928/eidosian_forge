import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
class TempEnv(gp.Env):

    def setParam(self, param, value):
        envparams[param] = value