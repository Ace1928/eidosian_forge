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
class LPAlgorithm(enum.Enum):
    """Selects an algorithm for solving linear programs.

    Attributes:
      * UNPSECIFIED: No algorithm is selected.
      * PRIMAL_SIMPLEX: The (primal) simplex method. Typically can provide primal
        and dual solutions, primal/dual rays on primal/dual unbounded problems,
        and a basis.
      * DUAL_SIMPLEX: The dual simplex method. Typically can provide primal and
        dual solutions, primal/dual rays on primal/dual unbounded problems, and a
        basis.
      * BARRIER: The barrier method, also commonly called an interior point method
        (IPM). Can typically give both primal and dual solutions. Some
        implementations can also produce rays on unbounded/infeasible problems. A
        basis is not given unless the underlying solver does "crossover" and
        finishes with simplex.
      * FIRST_ORDER: An algorithm based around a first-order method. These will
        typically produce both primal and dual solutions, and potentially also
        certificates of primal and/or dual infeasibility. First-order methods
        typically will provide solutions with lower accuracy, so users should take
        care to set solution quality parameters (e.g., tolerances) and to validate
        solutions.

    This must stay synchronized with math_opt_parameters_pb2.LPAlgorithmProto.
    """
    PRIMAL_SIMPLEX = math_opt_parameters_pb2.LP_ALGORITHM_PRIMAL_SIMPLEX
    DUAL_SIMPLEX = math_opt_parameters_pb2.LP_ALGORITHM_DUAL_SIMPLEX
    BARRIER = math_opt_parameters_pb2.LP_ALGORITHM_BARRIER
    FIRST_ORDER = math_opt_parameters_pb2.LP_ALGORITHM_FIRST_ORDER