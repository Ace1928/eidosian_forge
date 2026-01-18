import numpy as np
from scipy.linalg import (norm, get_lapack_funcs, solve_triangular,
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
Solve quadratic subproblem