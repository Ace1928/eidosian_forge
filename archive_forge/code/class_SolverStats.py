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
@dataclass
class SolverStats:
    """Reports some of the miscellaneous information that is returned
    by the solver after solving but that is not captured directly by
    the Problem instance.

    Attributes
    ----------
    solver_name : str
        The name of the solver.
    solve_time : double
        The time (in seconds) it took for the solver to solve the problem.
    setup_time : double
        The time (in seconds) it took for the solver to setup the problem.
    num_iters : int
        The number of iterations the solver had to go through to find a solution.
    extra_stats : object
        Extra statistics specific to the solver; these statistics are typically
        returned directly from the solver, without modification by CVXPY.
        This object may be a dict, or a custom Python object.
    """
    solver_name: str
    solve_time: Optional[float] = None
    setup_time: Optional[float] = None
    num_iters: Optional[int] = None
    extra_stats: Optional[dict] = None

    @classmethod
    def from_dict(cls, attr: dict, solver_name: str) -> 'SolverStats':
        """Construct a SolverStats object from a dictionary of attributes.

        Parameters
        ----------
        attr : dict
            A dictionary of attributes returned by the solver.
        solver_name : str
            The name of the solver.

        Returns
        -------
        SolverStats
            A SolverStats object.
        """
        return cls(solver_name, solve_time=attr.get(s.SOLVE_TIME), setup_time=attr.get(s.SETUP_TIME), num_iters=attr.get(s.NUM_ITERS), extra_stats=attr.get(s.EXTRA_STATS))