from __future__ import annotations
import time
import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import cvxpy.utilities as u
import cvxpy.utilities.performance_utils as perf
from cvxpy import Constant, error
from cvxpy import settings as s
from cvxpy.atoms.atom import Atom
from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from cvxpy.constraints.constraint import Constraint
from cvxpy.error import DPPError
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.variable import Variable
from cvxpy.interface.matrix_utilities import scalar_value
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.reductions import InverseData
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.dgp2dcp.dgp2dcp import Dgp2Dcp
from cvxpy.reductions.dqcp2dcp import dqcp2dcp
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solution import INF_OR_UNB_MESSAGE
from cvxpy.reductions.solvers import bisection
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import SOLVER_MAP_CONIC, SOLVER_MAP_QP
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.solvers.solving_chain import (
from cvxpy.settings import SOLVERS
from cvxpy.utilities import debug_tools
from cvxpy.utilities.deterministic import unique_list
def _add_custom_solver_candidates(self, custom_solver: Solver):
    """
        Returns a list of candidate solvers where custom_solver is the only potential option.

        Arguments
        ---------
        custom_solver : Solver

        Returns
        -------
        dict
            A dictionary of compatible solvers divided in `qp_solvers`
            and `conic_solvers`.

        Raises
        ------
        cvxpy.error.SolverError
            Raised if the name of the custom solver conflicts with the name of some officially
            supported solver
        """
    if custom_solver.name() in SOLVERS:
        message = 'Custom solvers must have a different name than the officially supported ones'
        raise error.SolverError(message)
    candidates = {'qp_solvers': [], 'conic_solvers': []}
    if not self.is_mixed_integer() or custom_solver.MIP_CAPABLE:
        if isinstance(custom_solver, QpSolver):
            SOLVER_MAP_QP[custom_solver.name()] = custom_solver
            candidates['qp_solvers'] = [custom_solver.name()]
        elif isinstance(custom_solver, ConicSolver):
            SOLVER_MAP_CONIC[custom_solver.name()] = custom_solver
            candidates['conic_solvers'] = [custom_solver.name()]
    return candidates