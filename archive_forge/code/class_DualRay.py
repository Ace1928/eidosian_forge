import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
@dataclasses.dataclass
class DualRay:
    """A direction of unbounded objective improvement in an optimization Model.

    A direction of unbounded improvement to the dual of an optimization,
    problem; equivalently, a certificate of primal infeasibility.

    E.g. consider the primal dual pair linear program pair:
      (Primal)\xa0 \xa0 \xa0 \xa0 \xa0 \xa0   (Dual)
      min c * x\xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0max b * y
      s.t. A * x >= b\xa0 \xa0 \xa0 \xa0s.t. y * A\xa0+ r = c
      x >= 0\xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0   y, r >= 0.

    The dual ray is the pair (y, r) satisfying:
      b * y > 0
      y * A + r = 0
      y, r >= 0.
    Observe that adding a positive multiple of (y, r) to dual feasible solution
    maintains dual feasibility and improves the objective (proving the dual is
    unbounded). The dual ray also proves the primal problem is infeasible.

    In the class DualRay below, y is dual_values and r is reduced_costs.

    For the general case, see go/mathopt-solutions and go/mathopt-dual (and note
    that the dual objective depends on r in the general case).

    Attributes:
      dual_values: The value assigned for each LinearConstraint in the model.
      reduced_costs: The value assigned for each Variable in the model.
    """
    dual_values: Dict[model.LinearConstraint, float] = dataclasses.field(default_factory=dict)
    reduced_costs: Dict[model.Variable, float] = dataclasses.field(default_factory=dict)