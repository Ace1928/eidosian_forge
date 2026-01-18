import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
class Elec:
    """Distribution of electrons on a sphere.

    Problem no 2 from COPS collection [2]_. Find
    the equilibrium state distribution (of minimal
    potential) of the electrons positioned on a
    conducting sphere.

    References
    ----------
    .. [1] E. D. Dolan, J. J. Mor'{e}, and T. S. Munson,
           "Benchmarking optimization software with COPS 3.0.",
            Argonne National Lab., Argonne, IL (US), 2004.
    """

    def __init__(self, n_electrons=200, random_state=0, constr_jac=None, constr_hess=None):
        self.n_electrons = n_electrons
        self.rng = np.random.RandomState(random_state)
        phi = self.rng.uniform(0, 2 * np.pi, self.n_electrons)
        theta = self.rng.uniform(-np.pi, np.pi, self.n_electrons)
        x = np.cos(theta) * np.cos(phi)
        y = np.cos(theta) * np.sin(phi)
        z = np.sin(theta)
        self.x0 = np.hstack((x, y, z))
        self.x_opt = None
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        self.bounds = None

    def _get_cordinates(self, x):
        x_coord = x[:self.n_electrons]
        y_coord = x[self.n_electrons:2 * self.n_electrons]
        z_coord = x[2 * self.n_electrons:]
        return (x_coord, y_coord, z_coord)

    def _compute_coordinate_deltas(self, x):
        x_coord, y_coord, z_coord = self._get_cordinates(x)
        dx = x_coord[:, None] - x_coord
        dy = y_coord[:, None] - y_coord
        dz = z_coord[:, None] - z_coord
        return (dx, dy, dz)

    def fun(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)
        with np.errstate(divide='ignore'):
            dm1 = (dx ** 2 + dy ** 2 + dz ** 2) ** (-0.5)
        dm1[np.diag_indices_from(dm1)] = 0
        return 0.5 * np.sum(dm1)

    def grad(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)
        with np.errstate(divide='ignore'):
            dm3 = (dx ** 2 + dy ** 2 + dz ** 2) ** (-1.5)
        dm3[np.diag_indices_from(dm3)] = 0
        grad_x = -np.sum(dx * dm3, axis=1)
        grad_y = -np.sum(dy * dm3, axis=1)
        grad_z = -np.sum(dz * dm3, axis=1)
        return np.hstack((grad_x, grad_y, grad_z))

    def hess(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)
        d = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        with np.errstate(divide='ignore'):
            dm3 = d ** (-3)
            dm5 = d ** (-5)
        i = np.arange(self.n_electrons)
        dm3[i, i] = 0
        dm5[i, i] = 0
        Hxx = dm3 - 3 * dx ** 2 * dm5
        Hxx[i, i] = -np.sum(Hxx, axis=1)
        Hxy = -3 * dx * dy * dm5
        Hxy[i, i] = -np.sum(Hxy, axis=1)
        Hxz = -3 * dx * dz * dm5
        Hxz[i, i] = -np.sum(Hxz, axis=1)
        Hyy = dm3 - 3 * dy ** 2 * dm5
        Hyy[i, i] = -np.sum(Hyy, axis=1)
        Hyz = -3 * dy * dz * dm5
        Hyz[i, i] = -np.sum(Hyz, axis=1)
        Hzz = dm3 - 3 * dz ** 2 * dm5
        Hzz[i, i] = -np.sum(Hzz, axis=1)
        H = np.vstack((np.hstack((Hxx, Hxy, Hxz)), np.hstack((Hxy, Hyy, Hyz)), np.hstack((Hxz, Hyz, Hzz))))
        return H

    @property
    def constr(self):

        def fun(x):
            x_coord, y_coord, z_coord = self._get_cordinates(x)
            return x_coord ** 2 + y_coord ** 2 + z_coord ** 2 - 1
        if self.constr_jac is None:

            def jac(x):
                x_coord, y_coord, z_coord = self._get_cordinates(x)
                Jx = 2 * np.diag(x_coord)
                Jy = 2 * np.diag(y_coord)
                Jz = 2 * np.diag(z_coord)
                return csc_matrix(np.hstack((Jx, Jy, Jz)))
        else:
            jac = self.constr_jac
        if self.constr_hess is None:

            def hess(x, v):
                D = 2 * np.diag(v)
                return block_diag(D, D, D)
        else:
            hess = self.constr_hess
        return NonlinearConstraint(fun, -np.inf, 0, jac, hess)