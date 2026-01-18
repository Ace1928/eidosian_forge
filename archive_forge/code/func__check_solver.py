import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def _check_solver(self, solver):
    ic = [1.0, 0.0]
    solver.set_initial_value(ic, 0.0)
    solver.integrate(pi)
    assert_array_almost_equal(solver.y, [-1.0, 0.0])