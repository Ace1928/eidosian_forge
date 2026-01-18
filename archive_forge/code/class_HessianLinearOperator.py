import time
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .._differentiable_functions import VectorFunction
from .._constraints import (
from .._hessian_update_strategy import BFGS
from .._optimize import OptimizeResult
from .._differentiable_functions import ScalarFunction
from .equality_constrained_sqp import equality_constrained_sqp
from .canonical_constraint import (CanonicalConstraint,
from .tr_interior_point import tr_interior_point
from .report import BasicReport, SQPReport, IPReport
class HessianLinearOperator:
    """Build LinearOperator from hessp"""

    def __init__(self, hessp, n):
        self.hessp = hessp
        self.n = n

    def __call__(self, x, *args):

        def matvec(p):
            return self.hessp(x, p, *args)
        return LinearOperator((self.n, self.n), matvec=matvec)