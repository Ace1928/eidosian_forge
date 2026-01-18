from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def actualResolve(self, lp, **kwargs):
    """
            looks at which variables have been modified and changes them
            """
    raise NotImplementedError('Resolves in CPLEX_PY not yet implemented')