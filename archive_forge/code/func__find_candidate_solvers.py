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
def _find_candidate_solvers(self, solver=None, gp: bool=False):
    """
        Find candidate solvers for the current problem. If solver
        is not None, it checks if the specified solver is compatible
        with the problem passed.

        Arguments
        ---------
        solver : Union[string, Solver, None]
            The name of the solver with which to solve the problem or an
            instance of a custom solver. If no solver is supplied
            (i.e., if solver is None), then the targeted solver may be any
            of those that are installed. If the problem is variable-free,
            then this parameter is ignored.
        gp : bool
            If True, the problem is parsed as a Disciplined Geometric Program
            instead of as a Disciplined Convex Program.

        Returns
        -------
        dict
            A dictionary of compatible solvers divided in `qp_solvers`
            and `conic_solvers`.

        Raises
        ------
        cvxpy.error.SolverError
            Raised if the problem is not DCP and `gp` is False.
        cvxpy.error.DGPError
            Raised if the problem is not DGP and `gp` is True.
        """
    candidates = {'qp_solvers': [], 'conic_solvers': []}
    if isinstance(solver, Solver):
        return self._add_custom_solver_candidates(solver)
    if solver is not None:
        if solver not in slv_def.INSTALLED_SOLVERS:
            raise error.SolverError('The solver %s is not installed.' % solver)
        if solver in slv_def.CONIC_SOLVERS:
            candidates['conic_solvers'] += [solver]
        if solver in slv_def.QP_SOLVERS:
            candidates['qp_solvers'] += [solver]
    else:
        candidates['qp_solvers'] = [s for s in slv_def.INSTALLED_SOLVERS if s in slv_def.QP_SOLVERS]
        candidates['conic_solvers'] = []
        for slv in slv_def.INSTALLED_SOLVERS:
            if slv in slv_def.CONIC_SOLVERS and slv != s.ECOS_BB:
                candidates['conic_solvers'].append(slv)
    if gp:
        if solver is not None and solver not in slv_def.CONIC_SOLVERS:
            raise error.SolverError("When `gp=True`, `solver` must be a conic solver (received '%s'); try calling " % solver + ' `solve()` with `solver=cvxpy.ECOS`.')
        elif solver is None:
            candidates['qp_solvers'] = []
    if self.is_mixed_integer():
        if slv_def.INSTALLED_MI_SOLVERS == [s.ECOS_BB] and solver != s.ECOS_BB:
            msg = "\n\n                    You need a mixed-integer solver for this model. Refer to the documentation\n                        https://www.cvxpy.org/tutorial/advanced/index.html#mixed-integer-programs\n                    for discussion on this topic.\n\n                    Quick fix 1: if you install the python package CVXOPT (pip install cvxopt),\n                    then CVXPY can use the open-source mixed-integer linear programming\n                    solver `GLPK`. If your problem is nonlinear then you can install SCIP\n                    (pip install pyscipopt).\n\n                    Quick fix 2: you can explicitly specify solver='ECOS_BB'. This may result\n                    in incorrect solutions and is not recommended.\n                "
            raise error.SolverError(msg)
        candidates['qp_solvers'] = [s for s in candidates['qp_solvers'] if slv_def.SOLVER_MAP_QP[s].MIP_CAPABLE]
        candidates['conic_solvers'] = [s for s in candidates['conic_solvers'] if slv_def.SOLVER_MAP_CONIC[s].MIP_CAPABLE]
        if not candidates['conic_solvers'] and (not candidates['qp_solvers']):
            raise error.SolverError('Problem is mixed-integer, but candidate QP/Conic solvers (%s) are not MIP-capable.' % (candidates['qp_solvers'] + candidates['conic_solvers']))
    return candidates