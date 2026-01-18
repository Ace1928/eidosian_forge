import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
class Pi(ODE):
    """Integrate 1/(t + 1j) from t=-10 to t=10"""
    stop_t = 20
    z0 = [0]
    cmplx = True

    def f(self, z, t):
        return array([1.0 / (t - 10 + 1j)])

    def verify(self, zs, t):
        u = -2j * np.arctan(10)
        return allclose(u, zs[-1, :], atol=self.atol, rtol=self.rtol)