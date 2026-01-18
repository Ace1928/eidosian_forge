import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def _run_solout_after_initial_test(self, integrator):
    ts = []
    ys = []
    t0 = 0.0
    tend = 10.0
    y0 = [1.0, 2.0]

    def solout(t, y):
        ts.append(t)
        ys.append(y.copy())

    def rhs(t, y):
        return [y[0] + y[1], -y[1] ** 2]
    ig = ode(rhs).set_integrator(integrator)
    ig.set_initial_value(y0, t0)
    ig.set_solout(solout)
    ret = ig.integrate(tend)
    assert_array_equal(ys[0], y0)
    assert_array_equal(ys[-1], ret)
    assert_equal(ts[0], t0)
    assert_equal(ts[-1], tend)