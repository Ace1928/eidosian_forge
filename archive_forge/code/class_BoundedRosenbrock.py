import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
class BoundedRosenbrock(Rosenbrock):
    """Rosenbrock subject to inequality constraints.

    The following optimization problem:
        minimize sum(100.0*(x[1] - x[0]**2)**2.0 + (1 - x[0])**2)
        subject to:  -2 <= x[0] <= 0
                      0 <= x[1] <= 2

    Taken from matlab ``fmincon`` documentation.
    """

    def __init__(self, random_state=0):
        Rosenbrock.__init__(self, 2, random_state)
        self.x0 = [-0.2, 0.2]
        self.x_opt = None
        self.bounds = Bounds([-2, 0], [0, 2])