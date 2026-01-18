from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
def _ppoly3d_eval(c, xs, xnew, ynew, znew, nu=None):
    """
    Straightforward evaluation of 3-D piecewise polynomial
    """
    if nu is None:
        nu = (0, 0, 0)
    out = np.empty((len(xnew),), dtype=c.dtype)
    nx, ny, nz = c.shape[:3]
    for jout, (x, y, z) in enumerate(zip(xnew, ynew, znew)):
        if not (xs[0][0] <= x <= xs[0][-1] and xs[1][0] <= y <= xs[1][-1] and (xs[2][0] <= z <= xs[2][-1])):
            out[jout] = np.nan
            continue
        j1 = np.searchsorted(xs[0], x) - 1
        j2 = np.searchsorted(xs[1], y) - 1
        j3 = np.searchsorted(xs[2], z) - 1
        s1 = x - xs[0][j1]
        s2 = y - xs[1][j2]
        s3 = z - xs[2][j3]
        val = 0
        for k1 in range(c.shape[0]):
            for k2 in range(c.shape[1]):
                for k3 in range(c.shape[2]):
                    val += c[nx - k1 - 1, ny - k2 - 1, nz - k3 - 1, j1, j2, j3] * _dpow(s1, k1, nu[0]) * _dpow(s2, k2, nu[1]) * _dpow(s3, k3, nu[2])
        out[jout] = val
    return out