import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
@dataclasses.dataclass
class DualSolution:
    """A solution to the dual of the optimization problem given by a Model.

    E.g. consider the primal dual pair linear program pair:
      (Primal)\xa0 \xa0 \xa0 \xa0 \xa0 \xa0   (Dual)
      min c * x\xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0max b * y
      s.t. A * x >= b\xa0 \xa0 \xa0 \xa0s.t. y * A\xa0+ r = c
      x >= 0\xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0   y, r >= 0.
    The dual solution is the pair (y, r). It is feasible if it satisfies the
    constraints from (Dual) above.

    Below, y is dual_values, r is reduced_costs, and b * y is objective_value.

    For the general case, see go/mathopt-solutions and go/mathopt-dual (and note
    that the dual objective depends on r in the general case).

    Attributes:
      dual_values: The value assigned for each LinearConstraint in the model.
      reduced_costs: The value assigned for each Variable in the model.
      objective_value: The value of the dual objective value at this solution.
        This value may not be always populated.
      feasibility_status: The feasibility of the solution as claimed by the
        solver.
    """
    dual_values: Dict[model.LinearConstraint, float] = dataclasses.field(default_factory=dict)
    reduced_costs: Dict[model.Variable, float] = dataclasses.field(default_factory=dict)
    objective_value: Optional[float] = None
    feasibility_status: SolutionStatus = SolutionStatus.UNDETERMINED

    def to_proto(self) -> solution_pb2.DualSolutionProto:
        """Returns an equivalent proto for a dual solution."""
        return solution_pb2.DualSolutionProto(dual_values=sparse_containers.to_sparse_double_vector_proto(self.dual_values), reduced_costs=sparse_containers.to_sparse_double_vector_proto(self.reduced_costs), objective_value=self.objective_value, feasibility_status=self.feasibility_status.value)