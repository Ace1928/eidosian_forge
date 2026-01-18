from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def actualSolve(self, lp):
    """Solve a well formulated lp problem"""
    raise PulpSolverError(f'CPLEX_PY: Not Available:\n{self.err}')