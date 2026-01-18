import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
@dataclasses.dataclass
class Termination:
    """An explanation of why the solver stopped.

    Attributes:
      reason: Why the solver stopped, e.g. it found a provably optimal solution.
        Additional information in `limit` when value is FEASIBLE or
        NO_SOLUTION_FOUND, see `limit` for details.
      limit: If the solver stopped early, what caused it to stop. Have value
        UNSPECIFIED when reason is not NO_SOLUTION_FOUND or FEASIBLE. May still be
        UNSPECIFIED when reason is NO_SOLUTION_FOUND or FEASIBLE, some solvers
        cannot fill this in.
      detail: Additional, information beyond reason about why the solver stopped,
        typically solver specific.
      problem_status: Feasibility statuses for primal and dual problems.
      objective_bounds: Bounds on the optimal objective value.
    """
    reason: TerminationReason = TerminationReason.OPTIMAL
    limit: Optional[Limit] = None
    detail: str = ''
    problem_status: ProblemStatus = ProblemStatus()
    objective_bounds: ObjectiveBounds = ObjectiveBounds()