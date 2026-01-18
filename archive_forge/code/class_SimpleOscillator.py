import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
class SimpleOscillator(ODE):
    """
    Free vibration of a simple oscillator::
        m \\ddot{u} + k u = 0, u(0) = u_0 \\dot{u}(0) \\dot{u}_0
    Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)
    """
    stop_t = 1 + 0.09
    z0 = array([1.0, 0.1], float)
    k = 4.0
    m = 1.0

    def f(self, z, t):
        tmp = zeros((2, 2), float)
        tmp[0, 1] = 1.0
        tmp[1, 0] = -self.k / self.m
        return dot(tmp, z)

    def verify(self, zs, t):
        omega = sqrt(self.k / self.m)
        u = self.z0[0] * cos(omega * t) + self.z0[1] * sin(omega * t) / omega
        return allclose(u, zs[:, 0], atol=self.atol, rtol=self.rtol)