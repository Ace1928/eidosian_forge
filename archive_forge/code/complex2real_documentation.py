from cvxpy import problems
from cvxpy import settings as s
from cvxpy.atoms.affine.upper_tri import vec_to_upper_tri
from cvxpy.constraints import (
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.lin_ops import lin_utils as lu
from cvxpy.reductions import InverseData, Solution
from cvxpy.reductions.complex2real.canonicalizers import CANON_METHODS as elim_cplx_methods
from cvxpy.reductions.reduction import Reduction
Lifts complex numbers to a real representation.