import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
class CPLEXShellWritePrioritiesFileKernel(CPLEXShellWritePrioritiesFile):
    suffix_cls = pmo.suffix

    @staticmethod
    def _set_suffix_value(suffix, variable, value):
        suffix[variable] = value

    def get_mock_model(self):
        model = pmo.block()
        model.x = pmo.variable(domain=Binary)
        model.con = pmo.constraint(expr=model.x >= 1)
        model.obj = pmo.objective(expr=model.x)
        return model