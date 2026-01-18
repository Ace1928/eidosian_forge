import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
class TestCumulativeSimpson:
    x0 = np.arange(4)
    y0 = x0 ** 2

    @pytest.mark.parametrize('use_dx', (False, True))
    @pytest.mark.parametrize('use_initial', (False, True))
    def test_1d(self, use_dx, use_initial):
        rng = np.random.default_rng(82456839535679456794)
        n = 10
        order = 3 if use_dx else 2
        dx = rng.random()
        x = np.sort(rng.random(n)) if order == 2 else np.arange(n) * dx + rng.random()
        i = np.arange(order + 1)[:, np.newaxis]
        c = rng.random(order + 1)[:, np.newaxis]
        y = np.sum(c * x ** i, axis=0)
        Y = np.sum(c * x ** (i + 1) / (i + 1), axis=0)
        ref = Y if use_initial else (Y - Y[0])[1:]
        initial = Y[0] if use_initial else None
        kwarg = {'dx': dx} if use_dx else {'x': x}
        res = cumulative_simpson(y, **kwarg, initial=initial)
        if not use_dx:
            assert_allclose(res, ref, rtol=2e-15)
        else:
            i0 = 0 if use_initial else 1
            assert_allclose(res, ref, rtol=0.0025)
            assert_allclose(res[i0::2], ref[i0::2], rtol=2e-15)

    @pytest.mark.parametrize('axis', np.arange(-3, 3))
    @pytest.mark.parametrize('x_ndim', (1, 3))
    @pytest.mark.parametrize('x_len', (1, 2, 7))
    @pytest.mark.parametrize('i_ndim', (None, 0, 3))
    @pytest.mark.parametrize('dx', (None, True))
    def test_nd(self, axis, x_ndim, x_len, i_ndim, dx):
        rng = np.random.default_rng(82456839535679456794)
        shape = [5, 6, x_len]
        shape[axis], shape[-1] = (shape[-1], shape[axis])
        shape_len_1 = shape.copy()
        shape_len_1[axis] = 1
        i_shape = shape_len_1 if i_ndim == 3 else ()
        y = rng.random(size=shape)
        x, dx = (None, None)
        if dx:
            dx = rng.random(size=shape_len_1) if x_ndim > 1 else rng.random()
        else:
            x = np.sort(rng.random(size=shape), axis=axis) if x_ndim > 1 else np.sort(rng.random(size=shape[axis]))
        initial = None if i_ndim is None else rng.random(size=i_shape)
        res = cumulative_simpson(y, x=x, dx=dx, initial=initial, axis=axis)
        ref = cumulative_simpson_nd_reference(y, x=x, dx=dx, initial=initial, axis=axis)
        np.testing.assert_allclose(res, ref, rtol=1e-15)

    @pytest.mark.parametrize(('message', 'kwarg_update'), [('x must be strictly increasing', dict(x=[2, 2, 3, 4])), ('x must be strictly increasing', dict(x=[x0, [2, 2, 4, 8]], y=[y0, y0])), ('x must be strictly increasing', dict(x=[x0, x0, x0], y=[y0, y0, y0], axis=0)), ('At least one point is required', dict(x=[], y=[])), ('`axis=4` is not valid for `y` with `y.ndim=1`', dict(axis=4)), ('shape of `x` must be the same as `y` or 1-D', dict(x=np.arange(5))), ('`initial` must either be a scalar or...', dict(initial=np.arange(5))), ('`dx` must either be a scalar or...', dict(x=None, dx=np.arange(5)))])
    def test_simpson_exceptions(self, message, kwarg_update):
        kwargs0 = dict(y=self.y0, x=self.x0, dx=None, initial=None, axis=-1)
        with pytest.raises(ValueError, match=message):
            cumulative_simpson(**dict(kwargs0, **kwarg_update))

    def test_special_cases(self):
        rng = np.random.default_rng(82456839535679456794)
        y = rng.random(size=10)
        res = cumulative_simpson(y, dx=0)
        assert_equal(res, 0)

    def _get_theoretical_diff_between_simps_and_cum_simps(self, y, x):
        """`cumulative_simpson` and `simpson` can be tested against other to verify
        they give consistent results. `simpson` will iteratively be called with 
        successively higher upper limits of integration. This function calculates
        the theoretical correction required to `simpson` at even intervals to match
        with `cumulative_simpson`.
        """
        d = np.diff(x, axis=-1)
        sub_integrals_h1 = _cumulative_simpson_unequal_intervals(y, d)
        sub_integrals_h2 = _cumulative_simpson_unequal_intervals(y[..., ::-1], d[..., ::-1])[..., ::-1]
        zeros_shape = (*y.shape[:-1], 1)
        theoretical_difference = np.concatenate([np.zeros(zeros_shape), sub_integrals_h1[..., 1:] - sub_integrals_h2[..., :-1], np.zeros(zeros_shape)], axis=-1)
        theoretical_difference[..., 1::2] = 0.0
        return theoretical_difference

    @given(y=hyp_num.arrays(np.float64, hyp_num.array_shapes(max_dims=4, min_side=3, max_side=10), elements=st.floats(-10, 10, allow_nan=False).filter(lambda x: abs(x) > 1e-07)))
    def test_cumulative_simpson_against_simpson_with_default_dx(self, y):
        """Theoretically, the output of `cumulative_simpson` will be identical
        to `simpson` at all even indices and in the last index. The first index
        will not match as `simpson` uses the trapezoidal rule when there are only two
        data points. Odd indices after the first index are shown to match with
        a mathematically-derived correction."""

        def simpson_reference(y):
            return np.stack([simpson(y[..., :i], dx=1.0) for i in range(2, y.shape[-1] + 1)], axis=-1)
        res = cumulative_simpson(y, dx=1.0)
        ref = simpson_reference(y)
        theoretical_difference = self._get_theoretical_diff_between_simps_and_cum_simps(y, x=np.arange(y.shape[-1]))
        np.testing.assert_allclose(res[..., 1:], ref[..., 1:] + theoretical_difference[..., 1:])

    @given(y=hyp_num.arrays(np.float64, hyp_num.array_shapes(max_dims=4, min_side=3, max_side=10), elements=st.floats(-10, 10, allow_nan=False).filter(lambda x: abs(x) > 1e-07)))
    def test_cumulative_simpson_against_simpson(self, y):
        """Theoretically, the output of `cumulative_simpson` will be identical
        to `simpson` at all even indices and in the last index. The first index
        will not match as `simpson` uses the trapezoidal rule when there are only two
        data points. Odd indices after the first index are shown to match with
        a mathematically-derived correction."""
        interval = 10 / (y.shape[-1] - 1)
        x = np.linspace(0, 10, num=y.shape[-1])
        x[1:] = x[1:] + 0.2 * interval * np.random.uniform(-1, 1, len(x) - 1)

        def simpson_reference(y, x):
            return np.stack([simpson(y[..., :i], x=x[..., :i]) for i in range(2, y.shape[-1] + 1)], axis=-1)
        res = cumulative_simpson(y, x=x)
        ref = simpson_reference(y, x)
        theoretical_difference = self._get_theoretical_diff_between_simps_and_cum_simps(y, x)
        np.testing.assert_allclose(res[..., 1:], ref[..., 1:] + theoretical_difference[..., 1:])