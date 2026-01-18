import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
class CoupledDecay(ODE):
    """
    3 coupled decays suited for banded treatment
    (banded mode makes it necessary when N>>3)
    """
    stiff = True
    stop_t = 0.5
    z0 = [5.0, 7.0, 13.0]
    lband = 1
    uband = 0
    lmbd = [0.17, 0.23, 0.29]

    def f(self, z, t):
        lmbd = self.lmbd
        return np.array([-lmbd[0] * z[0], -lmbd[1] * z[1] + lmbd[0] * z[0], -lmbd[2] * z[2] + lmbd[1] * z[1]])

    def jac(self, z, t):
        lmbd = self.lmbd
        j = np.zeros((self.lband + self.uband + 1, 3), order='F')

        def set_j(ri, ci, val):
            j[self.uband + ri - ci, ci] = val
        set_j(0, 0, -lmbd[0])
        set_j(1, 0, lmbd[0])
        set_j(1, 1, -lmbd[1])
        set_j(2, 1, lmbd[1])
        set_j(2, 2, -lmbd[2])
        return j

    def verify(self, zs, t):
        lmbd = np.array(self.lmbd)
        d10 = lmbd[1] - lmbd[0]
        d21 = lmbd[2] - lmbd[1]
        d20 = lmbd[2] - lmbd[0]
        e0 = np.exp(-lmbd[0] * t)
        e1 = np.exp(-lmbd[1] * t)
        e2 = np.exp(-lmbd[2] * t)
        u = np.vstack((self.z0[0] * e0, self.z0[1] * e1 + self.z0[0] * lmbd[0] / d10 * (e0 - e1), self.z0[2] * e2 + self.z0[1] * lmbd[1] / d21 * (e1 - e2) + lmbd[1] * lmbd[0] * self.z0[0] / d10 * (1 / d20 * (e0 - e2) - 1 / d21 * (e1 - e2)))).transpose()
        return allclose(u, zs, atol=self.atol, rtol=self.rtol)