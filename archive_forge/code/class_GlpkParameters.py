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
@dataclasses.dataclass
class GlpkParameters:
    """GLPK specific parameters for solving.

    Fields are optional to enable to capture user intention; if they set
    explicitly a value to then no generic solve parameters will overwrite this
    parameter. User specified solver specific parameters have priority on generic
    parameters.

    Attributes:
      compute_unbound_rays_if_possible: Compute the primal or dual unbound ray
        when the variable (structural or auxiliary) causing the unboundness is
        identified (see glp_get_unbnd_ray()). The unset value is equivalent to
        false. Rays are only available when solving linear programs, they are not
        available for MIPs. On top of that they are only available when using a
        simplex algorithm with the presolve disabled. A primal ray can only be
        built if the chosen LP algorithm is LPAlgorithm.PRIMAL_SIMPLEX. Same for a
        dual ray and LPAlgorithm.DUAL_SIMPLEX. The computation involves the basis
        factorization to be available which may lead to extra computations/errors.
    """
    compute_unbound_rays_if_possible: Optional[bool] = None

    def to_proto(self) -> glpk_pb2.GlpkParametersProto:
        return glpk_pb2.GlpkParametersProto(compute_unbound_rays_if_possible=self.compute_unbound_rays_if_possible)