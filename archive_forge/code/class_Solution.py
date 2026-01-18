import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
@dataclasses.dataclass
class Solution:
    """A solution to the optimization problem in a Model."""
    primal_solution: Optional[PrimalSolution] = None
    dual_solution: Optional[DualSolution] = None
    basis: Optional[Basis] = None

    def to_proto(self) -> solution_pb2.SolutionProto:
        """Returns an equivalent proto for a solution."""
        return solution_pb2.SolutionProto(primal_solution=self.primal_solution.to_proto() if self.primal_solution is not None else None, dual_solution=self.dual_solution.to_proto() if self.dual_solution is not None else None, basis=self.basis.to_proto() if self.basis is not None else None)