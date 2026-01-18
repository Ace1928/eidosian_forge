import logging
from pyomo.common.collections import ComponentMap
from pyomo.core import (
from pyomo.repn import generate_standard_repn
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
import pyomo.core.expr as EXPR
from pyomo.opt import ProblemSense
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.util.model_size import build_model_size_report
from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
import math
class GurobiPersistent4MindtPy(GurobiPersistent):
    """A new persistent interface to Gurobi.

    Args:
        GurobiPersistent (PersistentSolver): A class that provides a persistent interface to Gurobi.
    """

    def _intermediate_callback(self):

        def f(gurobi_model, where):
            """Callback function for Gurobi.

            Args:
                gurobi_model (Gurobi model): the Gurobi model derived from pyomo model.
                where (int): an enum member of gurobipy.GRB.Callback.
            """
            self._callback_func(self._pyomo_model, self, where, self.mindtpy_solver, self.config)
        return f