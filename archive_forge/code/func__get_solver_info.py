from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
def _get_solver_info(self):
    """
        Solver information dictionary

        Return:
        ------
        solver_status: a solver information dictionary containing the following key:value pairs
            -['square']: a string of square result solver status
            -['doe']: a string of doe result solver status
        """
    if self.result.solver.status == SolverStatus.ok and self.result.solver.termination_condition == TerminationCondition.optimal:
        self.status = 'converged'
    elif self.result.solver.termination_condition == TerminationCondition.infeasible:
        self.status = 'infeasible'
    else:
        self.status = self.result.solver.status