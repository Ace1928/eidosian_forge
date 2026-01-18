from __future__ import annotations
import warnings
import numpy as np
from cvxpy.atoms import EXP_ATOMS, NONPOS_ATOMS, PSD_ATOMS, SOC_ATOMS
from cvxpy.constraints import (
from cvxpy.constraints.exponential import OpRelEntrConeQuad, RelEntrConeQuad
from cvxpy.error import DCPError, DGPError, DPPError, SolverError
from cvxpy.problems.objective import Maximize
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.complex2real import complex2real
from cvxpy.reductions.cone2cone.approximations import APPROX_CONES, QuadApprox
from cvxpy.reductions.cone2cone.exotic2common import (
from cvxpy.reductions.cone2cone.soc2psd import SOC2PSD
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.dgp2dcp.dgp2dcp import Dgp2Dcp
from cvxpy.reductions.discrete2mixedint.valinvec2mixedint import (
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.qp2quad_form import qp2symbolic_qp
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.reductions.solvers.constant_solver import ConstantSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.settings import ECOS, PARAM_THRESHOLD
from cvxpy.utilities.debug_tools import build_non_disciplined_error_msg
class SolvingChain(Chain):
    """A reduction chain that ends with a solver.

    Parameters
    ----------
    reductions : list[Reduction]
        A list of reductions. The last reduction in the list must be a solver
        instance.

    Attributes
    ----------
    reductions : list[Reduction]
        A list of reductions.
    solver : Solver
        The solver, i.e., reductions[-1].
    """

    def __init__(self, problem=None, reductions=None) -> None:
        super(SolvingChain, self).__init__(problem=problem, reductions=reductions)
        if not isinstance(self.reductions[-1], Solver):
            raise ValueError('Solving chains must terminate with a Solver.')
        self.solver = self.reductions[-1]

    def prepend(self, chain) -> 'SolvingChain':
        """
        Create and return a new SolvingChain by concatenating
        chain with this instance.
        """
        return SolvingChain(reductions=chain.reductions + self.reductions)

    def solve(self, problem, warm_start: bool, verbose: bool, solver_opts):
        """Solves the problem by applying the chain.

        Applies each reduction in the chain to the problem, solves it,
        and then inverts the chain to return a solution of the supplied
        problem.

        Parameters
        ----------
        problem : Problem
            The problem to solve.
        warm_start : bool
            Whether to warm start the solver.
        verbose : bool
            Whether to enable solver verbosity.
        solver_opts : dict
            Solver specific options.

        Returns
        -------
        solution : Solution
            A solution to the problem.
        """
        data, inverse_data = self.apply(problem)
        solution = self.solver.solve_via_data(data, warm_start, verbose, solver_opts)
        return self.invert(solution, inverse_data)

    def solve_via_data(self, problem, data, warm_start: bool=False, verbose: bool=False, solver_opts={}):
        """Solves the problem using the data output by the an apply invocation.

        The semantics are:

        .. code :: python

            data, inverse_data = solving_chain.apply(problem)
            solution = solving_chain.invert(solver_chain.solve_via_data(data, ...))

        which is equivalent to writing

        .. code :: python

            solution = solving_chain.solve(problem, ...)

        Parameters
        ----------
        problem : Problem
            The problem to solve.
        data : map
            Data for the solver.
        warm_start : bool
            Whether to warm start the solver.
        verbose : bool
            Whether to enable solver verbosity.
        solver_opts : dict
            Solver specific options.

        Returns
        -------
        raw solver solution
            The information returned by the solver; this is not necessarily
            a Solution object.
        """
        return self.solver.solve_via_data(data, warm_start, verbose, solver_opts, problem._solver_cache)