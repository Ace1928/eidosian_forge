import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
@dataclasses.dataclass(frozen=True)
class ProblemStatus:
    """Feasibility status of the primal problem and its dual (or dual relaxation).

    Statuses are as claimed by the solver and a dual relaxation is the dual of a
    continuous relaxation for the original problem (e.g. the LP relaxation of a
    MIP). The solver is not required to return a certificate for the feasibility
    or infeasibility claims (e.g. the solver may claim primal feasibility without
    returning a primal feasible solutuion). This combined status gives a
    comprehensive description of a solver's claims about feasibility and
    unboundedness of the solved problem. For instance,
      * a feasible status for primal and dual problems indicates the primal is
        feasible and bounded and likely has an optimal solution (guaranteed for
        problems without non-linear constraints).
      * a primal feasible and a dual infeasible status indicates the primal
        problem is unbounded (i.e. has arbitrarily good solutions).
    Note that a dual infeasible status by itself (i.e. accompanied by an
    undetermined primal status) does not imply the primal problem is unbounded as
    we could have both problems be infeasible. Also, while a primal and dual
    feasible status may imply the existence of an optimal solution, it does not
    guarantee the solver has actually found such optimal solution.

    Attributes:
      primal_status: Status for the primal problem.
      dual_status: Status for the dual problem (or for the dual of a continuous
        relaxation).
      primal_or_dual_infeasible: If true, the solver claims the primal or dual
        problem is infeasible, but it does not know which (or if both are
        infeasible). Can be true only when primal_problem_status =
        dual_problem_status = kUndetermined. This extra information is often
        needed when preprocessing determines there is no optimal solution to the
        problem (but can't determine if it is due to infeasibility, unboundedness,
        or both).
    """
    primal_status: FeasibilityStatus = FeasibilityStatus.UNDETERMINED
    dual_status: FeasibilityStatus = FeasibilityStatus.UNDETERMINED
    primal_or_dual_infeasible: bool = False

    def to_proto(self) -> result_pb2.ProblemStatusProto:
        """Returns an equivalent proto for a problem status."""
        return result_pb2.ProblemStatusProto(primal_status=self.primal_status.value, dual_status=self.dual_status.value, primal_or_dual_infeasible=self.primal_or_dual_infeasible)