from cvxpy.atoms import abs, max, maximum, norm1, norm_inf, sum_largest
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.eliminate_pwl.canonicalizers import CANON_METHODS as elim_pwl_methods
Eliminates piecewise linear atoms.