import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
class HyperbolicIneq:
    """Problem 15.1 from Nocedal and Wright

    The following optimization problem:
        minimize 1/2*(x[0] - 2)**2 + 1/2*(x[1] - 1/2)**2
        Subject to: 1/(x[0] + 1) - x[1] >= 1/4
                                   x[0] >= 0
                                   x[1] >= 0
    """

    def __init__(self, constr_jac=None, constr_hess=None):
        self.x0 = [0, 0]
        self.x_opt = [1.952823, 0.088659]
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        self.bounds = Bounds(0, np.inf)

    def fun(self, x):
        return 1 / 2 * (x[0] - 2) ** 2 + 1 / 2 * (x[1] - 1 / 2) ** 2

    def grad(self, x):
        return [x[0] - 2, x[1] - 1 / 2]

    def hess(self, x):
        return np.eye(2)

    @property
    def constr(self):

        def fun(x):
            return 1 / (x[0] + 1) - x[1]
        if self.constr_jac is None:

            def jac(x):
                return [[-1 / (x[0] + 1) ** 2, -1]]
        else:
            jac = self.constr_jac
        if self.constr_hess is None:

            def hess(x, v):
                return 2 * v[0] * np.array([[1 / (x[0] + 1) ** 3, 0], [0, 0]])
        else:
            hess = self.constr_hess
        return NonlinearConstraint(fun, 0.25, np.inf, jac, hess)