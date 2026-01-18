import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
class TestNdBSpline:

    def test_1D(self):
        rng = np.random.default_rng(12345)
        n, k = (11, 3)
        n_tr = 7
        t = np.sort(rng.uniform(size=n + k + 1))
        c = rng.uniform(size=(n, n_tr))
        b = BSpline(t, c, k)
        nb = NdBSpline((t,), c, k)
        xi = rng.uniform(size=21)
        assert_allclose(nb(xi[:, None]), b(xi), atol=1e-14)
        assert nb(xi[:, None]).shape == (xi.shape[0], c.shape[1])

    def make_2d_case(self):
        x = np.arange(6)
        y = x ** 3
        spl = make_interp_spline(x, y, k=3)
        y_1 = x ** 3 + 2 * x
        spl_1 = make_interp_spline(x, y_1, k=3)
        t2 = (spl.t, spl_1.t)
        c2 = spl.c[:, None] * spl_1.c[None, :]
        return (t2, c2, 3)

    def make_2d_mixed(self):
        x = np.arange(6)
        y = x ** 3
        spl = make_interp_spline(x, y, k=3)
        x = np.arange(5) + 1.5
        y_1 = x ** 2 + 2 * x
        spl_1 = make_interp_spline(x, y_1, k=2)
        t2 = (spl.t, spl_1.t)
        c2 = spl.c[:, None] * spl_1.c[None, :]
        return (t2, c2, spl.k, spl_1.k)

    def test_2D_separable(self):
        xi = [(1.5, 2.5), (2.5, 1), (0.5, 1.5)]
        t2, c2, k = self.make_2d_case()
        target = [x ** 3 * (y ** 3 + 2 * y) for x, y in xi]
        assert_allclose([bspline2(xy, t2, c2, k) for xy in xi], target, atol=1e-14)
        bspl2 = NdBSpline(t2, c2, k=3)
        assert bspl2(xi).shape == (len(xi),)
        assert_allclose(bspl2(xi), target, atol=1e-14)
        rng = np.random.default_rng(12345)
        xi = rng.uniform(size=(4, 3, 2)) * 5
        result = bspl2(xi)
        assert result.shape == (4, 3)
        x, y = xi.reshape((-1, 2)).T
        assert_allclose(result.ravel(), x ** 3 * (y ** 3 + 2 * y), atol=1e-14)

    def test_2D_separable_2(self):
        ndim = 2
        xi = [(1.5, 2.5), (2.5, 1), (0.5, 1.5)]
        target = [x ** 3 * (y ** 3 + 2 * y) for x, y in xi]
        t2, c2, k = self.make_2d_case()
        c2_4 = np.dstack((c2, c2, c2, c2))
        xy = (1.5, 2.5)
        bspl2_4 = NdBSpline(t2, c2_4, k=3)
        result = bspl2_4(xy)
        val_single = NdBSpline(t2, c2, k)(xy)
        assert result.shape == (4,)
        assert_allclose(result, [val_single] * 4, atol=1e-14)
        assert bspl2_4(xi).shape == np.shape(xi)[:-1] + bspl2_4.c.shape[ndim:]
        assert_allclose(bspl2_4(xi) - np.asarray(target)[:, None], 0, atol=5e-14)
        c2_22 = c2_4.reshape((6, 6, 2, 2))
        bspl2_22 = NdBSpline(t2, c2_22, k=3)
        result = bspl2_22(xy)
        assert result.shape == (2, 2)
        assert_allclose(result, [[val_single, val_single], [val_single, val_single]], atol=1e-14)
        assert bspl2_22(xi).shape == np.shape(xi)[:-1] + bspl2_22.c.shape[ndim:]
        assert_allclose(bspl2_22(xi) - np.asarray(target)[:, None, None], 0, atol=5e-14)

    def test_2D_random(self):
        rng = np.random.default_rng(12345)
        k = 3
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        c = rng.uniform(size=(tx.size - k - 1, ty.size - k - 1))
        spl = NdBSpline((tx, ty), c, k=k)
        xi = (1.0, 1.0)
        assert_allclose(spl(xi), bspline2(xi, (tx, ty), c, k), atol=1e-14)
        xi = np.c_[[1, 1.5, 2], [1.1, 1.6, 2.1]]
        assert_allclose(spl(xi), [bspline2(xy, (tx, ty), c, k) for xy in xi], atol=1e-14)

    def test_2D_mixed(self):
        t2, c2, kx, ky = self.make_2d_mixed()
        xi = [(1.4, 4.5), (2.5, 2.4), (4.5, 3.5)]
        target = [x ** 3 * (y ** 2 + 2 * y) for x, y in xi]
        bspl2 = NdBSpline(t2, c2, k=(kx, ky))
        assert bspl2(xi).shape == (len(xi),)
        assert_allclose(bspl2(xi), target, atol=1e-14)

    def test_2D_derivative(self):
        t2, c2, kx, ky = self.make_2d_mixed()
        xi = [(1.4, 4.5), (2.5, 2.4), (4.5, 3.5)]
        bspl2 = NdBSpline(t2, c2, k=(kx, ky))
        der = bspl2(xi, nu=(1, 0))
        assert_allclose(der, [3 * x ** 2 * (y ** 2 + 2 * y) for x, y in xi], atol=1e-14)
        der = bspl2(xi, nu=(1, 1))
        assert_allclose(der, [3 * x ** 2 * (2 * y + 2) for x, y in xi], atol=1e-14)
        der = bspl2(xi, nu=(0, 0))
        assert_allclose(der, [x ** 3 * (y ** 2 + 2 * y) for x, y in xi], atol=1e-14)

    def test_2D_mixed_random(self):
        rng = np.random.default_rng(12345)
        kx, ky = (2, 3)
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        c = rng.uniform(size=(tx.size - kx - 1, ty.size - ky - 1))
        xi = np.c_[[1, 1.5, 2], [1.1, 1.6, 2.1]]
        bspl2 = NdBSpline((tx, ty), c, k=(kx, ky))
        bspl2_0 = NdBSpline0((tx, ty), c, k=(kx, ky))
        assert_allclose(bspl2(xi), [bspl2_0(xp) for xp in xi], atol=1e-14)

    def test_tx_neq_ty(self):
        x = np.arange(6)
        y = np.arange(7) + 1.5
        spl_x = make_interp_spline(x, x ** 3, k=3)
        spl_y = make_interp_spline(y, y ** 2 + 2 * y, k=3)
        cc = spl_x.c[:, None] * spl_y.c[None, :]
        bspl = NdBSpline((spl_x.t, spl_y.t), cc, (spl_x.k, spl_y.k))
        values = (x ** 3)[:, None] * (y ** 2 + 2 * y)[None, :]
        rgi = RegularGridInterpolator((x, y), values)
        xi = [(a, b) for a, b in itertools.product(x, y)]
        bxi = bspl(xi)
        assert not np.isnan(bxi).any()
        assert_allclose(bxi, rgi(xi), atol=1e-14)
        assert_allclose(bxi.reshape(values.shape), values, atol=1e-14)

    def make_3d_case(self):
        x = np.arange(6)
        y = x ** 3
        spl = make_interp_spline(x, y, k=3)
        y_1 = x ** 3 + 2 * x
        spl_1 = make_interp_spline(x, y_1, k=3)
        y_2 = x ** 3 + 3 * x + 1
        spl_2 = make_interp_spline(x, y_2, k=3)
        t2 = (spl.t, spl_1.t, spl_2.t)
        c2 = spl.c[:, None, None] * spl_1.c[None, :, None] * spl_2.c[None, None, :]
        return (t2, c2, 3)

    def test_3D_separable(self):
        rng = np.random.default_rng(12345)
        x, y, z = rng.uniform(size=(3, 11)) * 5
        target = x ** 3 * (y ** 3 + 2 * y) * (z ** 3 + 3 * z + 1)
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)
        xi = [_ for _ in zip(x, y, z)]
        result = bspl3(xi)
        assert result.shape == (11,)
        assert_allclose(result, target, atol=1e-14)

    def test_3D_derivative(self):
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)
        rng = np.random.default_rng(12345)
        x, y, z = rng.uniform(size=(3, 11)) * 5
        xi = [_ for _ in zip(x, y, z)]
        assert_allclose(bspl3(xi, nu=(1, 0, 0)), 3 * x ** 2 * (y ** 3 + 2 * y) * (z ** 3 + 3 * z + 1), atol=1e-14)
        assert_allclose(bspl3(xi, nu=(2, 0, 0)), 6 * x * (y ** 3 + 2 * y) * (z ** 3 + 3 * z + 1), atol=1e-14)
        assert_allclose(bspl3(xi, nu=(2, 1, 0)), 6 * x * (3 * y ** 2 + 2) * (z ** 3 + 3 * z + 1), atol=1e-14)
        assert_allclose(bspl3(xi, nu=(2, 1, 3)), 6 * x * (3 * y ** 2 + 2) * 6, atol=1e-14)
        assert_allclose(bspl3(xi, nu=(2, 1, 4)), np.zeros(len(xi)), atol=1e-14)

    def test_3D_random(self):
        rng = np.random.default_rng(12345)
        k = 3
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        tz = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        c = rng.uniform(size=(tx.size - k - 1, ty.size - k - 1, tz.size - k - 1))
        spl = NdBSpline((tx, ty, tz), c, k=k)
        spl_0 = NdBSpline0((tx, ty, tz), c, k=k)
        xi = (1.0, 1.0, 1)
        assert_allclose(spl(xi), spl_0(xi), atol=1e-14)
        xi = np.c_[[1, 1.5, 2], [1.1, 1.6, 2.1], [0.9, 1.4, 1.9]]
        assert_allclose(spl(xi), [spl_0(xp) for xp in xi], atol=1e-14)

    def test_3D_random_complex(self):
        rng = np.random.default_rng(12345)
        k = 3
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        tz = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        c = rng.uniform(size=(tx.size - k - 1, ty.size - k - 1, tz.size - k - 1)) + rng.uniform(size=(tx.size - k - 1, ty.size - k - 1, tz.size - k - 1)) * 1j
        spl = NdBSpline((tx, ty, tz), c, k=k)
        spl_re = NdBSpline((tx, ty, tz), c.real, k=k)
        spl_im = NdBSpline((tx, ty, tz), c.imag, k=k)
        xi = np.c_[[1, 1.5, 2], [1.1, 1.6, 2.1], [0.9, 1.4, 1.9]]
        assert_allclose(spl(xi), spl_re(xi) + 1j * spl_im(xi), atol=1e-14)

    @pytest.mark.parametrize('cls_extrap', [None, True])
    @pytest.mark.parametrize('call_extrap', [None, True])
    def test_extrapolate_3D_separable(self, cls_extrap, call_extrap):
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3, extrapolate=cls_extrap)
        x, y, z = ([-2, -1, 7], [-3, -0.5, 6.5], [-1, -1.5, 7.5])
        x, y, z = map(np.asarray, (x, y, z))
        xi = [_ for _ in zip(x, y, z)]
        target = x ** 3 * (y ** 3 + 2 * y) * (z ** 3 + 3 * z + 1)
        result = bspl3(xi, extrapolate=call_extrap)
        assert_allclose(result, target, atol=1e-14)

    @pytest.mark.parametrize('extrap', [(False, True), (True, None)])
    def test_extrapolate_3D_separable_2(self, extrap):
        t3, c3, k = self.make_3d_case()
        cls_extrap, call_extrap = extrap
        bspl3 = NdBSpline(t3, c3, k=3, extrapolate=cls_extrap)
        x, y, z = ([-2, -1, 7], [-3, -0.5, 6.5], [-1, -1.5, 7.5])
        x, y, z = map(np.asarray, (x, y, z))
        xi = [_ for _ in zip(x, y, z)]
        target = x ** 3 * (y ** 3 + 2 * y) * (z ** 3 + 3 * z + 1)
        result = bspl3(xi, extrapolate=call_extrap)
        assert_allclose(result, target, atol=1e-14)

    def test_extrapolate_false_3D_separable(self):
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)
        x, y, z = ([-2, 1, 7], [-3, 0.5, 6.5], [-1, 1.5, 7.5])
        x, y, z = map(np.asarray, (x, y, z))
        xi = [_ for _ in zip(x, y, z)]
        target = x ** 3 * (y ** 3 + 2 * y) * (z ** 3 + 3 * z + 1)
        result = bspl3(xi, extrapolate=False)
        assert np.isnan(result[0])
        assert np.isnan(result[-1])
        assert_allclose(result[1:-1], target[1:-1], atol=1e-14)

    def test_x_nan_3D(self):
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)
        x = np.asarray([-2, 3, np.nan, 1, 2, 7, np.nan])
        y = np.asarray([-3, 3.5, 1, np.nan, 3, 6.5, 6.5])
        z = np.asarray([-1, 3.5, 2, 3, np.nan, 7.5, 7.5])
        xi = [_ for _ in zip(x, y, z)]
        target = x ** 3 * (y ** 3 + 2 * y) * (z ** 3 + 3 * z + 1)
        mask = np.isnan(x) | np.isnan(y) | np.isnan(z)
        target[mask] = np.nan
        result = bspl3(xi)
        assert np.isnan(result[mask]).all()
        assert_allclose(result, target, atol=1e-14)

    def test_non_c_contiguous(self):
        rng = np.random.default_rng(12345)
        kx, ky = (3, 3)
        tx = np.sort(rng.uniform(low=0, high=4, size=16))
        tx = np.r_[(tx[0],) * kx, tx, (tx[-1],) * kx]
        ty = np.sort(rng.uniform(low=0, high=4, size=16))
        ty = np.r_[(ty[0],) * ky, ty, (ty[-1],) * ky]
        assert not tx[::2].flags.c_contiguous
        assert not ty[::2].flags.c_contiguous
        c = rng.uniform(size=(tx.size // 2 - kx - 1, ty.size // 2 - ky - 1))
        c = c.T
        assert not c.flags.c_contiguous
        xi = np.c_[[1, 1.5, 2], [1.1, 1.6, 2.1]]
        bspl2 = NdBSpline((tx[::2], ty[::2]), c, k=(kx, ky))
        bspl2_0 = NdBSpline0((tx[::2], ty[::2]), c, k=(kx, ky))
        assert_allclose(bspl2(xi), [bspl2_0(xp) for xp in xi], atol=1e-14)

    def test_readonly(self):
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)
        for i in range(3):
            t3[i].flags.writeable = False
        c3.flags.writeable = False
        bspl3_ = NdBSpline(t3, c3, k=3)
        assert bspl3((1, 2, 3)) == bspl3_((1, 2, 3))