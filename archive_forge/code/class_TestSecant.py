from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
class TestSecant:
    """Check that some Jacobian approximations satisfy the secant condition"""
    xs = [np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 3.0, 4.0, 5.0, 1.0]), np.array([3.0, 4.0, 5.0, 1.0, 2.0]), np.array([4.0, 5.0, 1.0, 2.0, 3.0]), np.array([9.0, 1.0, 9.0, 1.0, 3.0]), np.array([0.0, 1.0, 9.0, 1.0, 3.0]), np.array([5.0, 5.0, 7.0, 1.0, 1.0]), np.array([1.0, 2.0, 7.0, 5.0, 1.0])]
    fs = [x ** 2 - 1 for x in xs]

    def _check_secant(self, jac_cls, npoints=1, **kw):
        """
        Check that the given Jacobian approximation satisfies secant
        conditions for last `npoints` points.
        """
        jac = jac_cls(**kw)
        jac.setup(self.xs[0], self.fs[0], None)
        for j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
            jac.update(x, f)
            for k in range(min(npoints, j + 1)):
                dx = self.xs[j - k + 1] - self.xs[j - k]
                df = self.fs[j - k + 1] - self.fs[j - k]
                assert_(np.allclose(dx, jac.solve(df)))
            if j >= npoints:
                dx = self.xs[j - npoints + 1] - self.xs[j - npoints]
                df = self.fs[j - npoints + 1] - self.fs[j - npoints]
                assert_(not np.allclose(dx, jac.solve(df)))

    def test_broyden1(self):
        self._check_secant(nonlin.BroydenFirst)

    def test_broyden2(self):
        self._check_secant(nonlin.BroydenSecond)

    def test_broyden1_update(self):
        jac = nonlin.BroydenFirst(alpha=0.1)
        jac.setup(self.xs[0], self.fs[0], None)
        B = np.identity(5) * (-1 / 0.1)
        for last_j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
            df = f - self.fs[last_j]
            dx = x - self.xs[last_j]
            B += (df - dot(B, dx))[:, None] * dx[None, :] / dot(dx, dx)
            jac.update(x, f)
            assert_(np.allclose(jac.todense(), B, rtol=1e-10, atol=1e-13))

    def test_broyden2_update(self):
        jac = nonlin.BroydenSecond(alpha=0.1)
        jac.setup(self.xs[0], self.fs[0], None)
        H = np.identity(5) * -0.1
        for last_j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
            df = f - self.fs[last_j]
            dx = x - self.xs[last_j]
            H += (dx - dot(H, df))[:, None] * df[None, :] / dot(df, df)
            jac.update(x, f)
            assert_(np.allclose(jac.todense(), inv(H), rtol=1e-10, atol=1e-13))

    def test_anderson(self):
        self._check_secant(nonlin.Anderson, M=3, w0=0, npoints=3)