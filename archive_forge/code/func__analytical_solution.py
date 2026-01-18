import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import ode
def _analytical_solution(a, y0, t):
    """
    Analytical solution to the linear differential equations dy/dt = a*y.

    The solution is only valid if `a` is diagonalizable.

    Returns a 2-D array with shape (len(t), len(y0)).
    """
    lam, v = np.linalg.eig(a)
    c = np.linalg.solve(v, y0)
    e = c * np.exp(lam * t.reshape(-1, 1))
    sol = e.dot(v.T)
    return sol