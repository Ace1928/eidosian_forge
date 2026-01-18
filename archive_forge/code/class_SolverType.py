import dataclasses
import datetime
import enum
from typing import Dict, Optional
from ortools.pdlp import solvers_pb2 as pdlp_solvers_pb2
from ortools.glop import parameters_pb2 as glop_parameters_pb2
from ortools.gscip import gscip_pb2
from ortools.math_opt import parameters_pb2 as math_opt_parameters_pb2
from ortools.math_opt.solvers import glpk_pb2
from ortools.math_opt.solvers import gurobi_pb2
from ortools.math_opt.solvers import highs_pb2
from ortools.math_opt.solvers import osqp_pb2
from ortools.sat import sat_parameters_pb2
@enum.unique
class SolverType(enum.Enum):
    """The underlying solver to use.

    This must stay synchronized with math_opt_parameters_pb2.SolverTypeProto.

    Attributes:
      GSCIP: Solving Constraint Integer Programs (SCIP) solver (third party).
        Supports LP, MIP, and nonconvex integer quadratic problems. No dual data
        for LPs is returned though. Prefer GLOP for LPs.
      GUROBI: Gurobi solver (third party). Supports LP, MIP, and nonconvex integer
        quadratic problems. Generally the fastest option, but has special
        licensing, see go/gurobi-google for details.
      GLOP: Google's Glop linear solver. Supports LP with primal and dual simplex
        methods.
      CP_SAT: Google's CP-SAT solver. Supports problems where all variables are
        integer and bounded (or implied to be after presolve). Experimental
        support to rescale and discretize problems with continuous variables.
      MOE:begin_intracomment_strip
      PDLP: Google's PDLP solver. Supports LP and convex diagonal quadratic
        objectives. Uses first order methods rather than simplex. Can solve very
        large problems.
      MOE:end_intracomment_strip
      GLPK: GNU Linear Programming Kit (GLPK) (third party). Supports MIP and LP.
        Thread-safety: GLPK use thread-local storage for memory allocations. As a
        consequence when using IncrementalSolver, the user must make sure that
        instances are closed on the same thread as they are created or GLPK will
        crash. To do so, use `with` or call IncrementalSolver#close(). It seems OK
        to call IncrementalSolver#Solve() from another thread than the one used to
        create the Solver but it is not documented by GLPK and should be avoided.
        Of course these limitations do not apply to the solve() function that
        recreates a new GLPK problem in the calling thread and destroys before
        returning.  When solving a LP with the presolver, a solution (and the
        unbound rays) are only returned if an optimal solution has been found.
        Else nothing is returned. See glpk-5.0/doc/glpk.pdf page #40 available
        from glpk-5.0.tar.gz for details.
      OSQP: The Operator Splitting Quadratic Program (OSQP) solver (third party).
        Supports continuous problems with linear constraints and linear or convex
        quadratic objectives. Uses a first-order method.
      ECOS: The Embedded Conic Solver (ECOS) (third party). Supports LP and SOCP
        problems. Uses interior point methods (barrier).
      SCS: The Splitting Conic Solver (SCS) (third party). Supports LP and SOCP
        problems. Uses a first-order method.
      HIGHS: The HiGHS Solver (third party). Supports LP and MIP problems (convex
        QPs are unimplemented).
      SANTORINI: The Santorini Solver (first party). Supports MIP. Experimental,
        do not use in production.
    """
    GSCIP = math_opt_parameters_pb2.SOLVER_TYPE_GSCIP
    GUROBI = math_opt_parameters_pb2.SOLVER_TYPE_GUROBI
    GLOP = math_opt_parameters_pb2.SOLVER_TYPE_GLOP
    CP_SAT = math_opt_parameters_pb2.SOLVER_TYPE_CP_SAT
    PDLP = math_opt_parameters_pb2.SOLVER_TYPE_PDLP
    GLPK = math_opt_parameters_pb2.SOLVER_TYPE_GLPK
    OSQP = math_opt_parameters_pb2.SOLVER_TYPE_OSQP
    ECOS = math_opt_parameters_pb2.SOLVER_TYPE_ECOS
    SCS = math_opt_parameters_pb2.SOLVER_TYPE_SCS
    HIGHS = math_opt_parameters_pb2.SOLVER_TYPE_HIGHS
    SANTORINI = math_opt_parameters_pb2.SOLVER_TYPE_SANTORINI