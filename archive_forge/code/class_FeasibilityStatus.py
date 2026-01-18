import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
@enum.unique
class FeasibilityStatus(enum.Enum):
    """Problem feasibility status as claimed by the solver.

      (solver is not required to return a certificate for the claim.)

    Attributes:
      UNDETERMINED: Solver does not claim a status.
      FEASIBLE: Solver claims the problem is feasible.
      INFEASIBLE: Solver claims the problem is infeasible.
    """
    UNDETERMINED = result_pb2.FEASIBILITY_STATUS_UNDETERMINED
    FEASIBLE = result_pb2.FEASIBILITY_STATUS_FEASIBLE
    INFEASIBLE = result_pb2.FEASIBILITY_STATUS_INFEASIBLE