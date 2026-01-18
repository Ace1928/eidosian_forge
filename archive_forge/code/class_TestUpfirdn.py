import numpy as np
from itertools import product
from numpy.testing import assert_equal, assert_allclose
from pytest import raises as assert_raises
import pytest
from scipy.signal import upfirdn, firwin
from scipy.signal._upfirdn import _output_len, _upfirdn_modes
from scipy.signal._upfirdn_apply import _pad_test
class TestUpfirdn:

    def test_valid_input(self):
        assert_raises(ValueError, upfirdn, [1], [1], 1, 0)
        assert_raises(ValueError, upfirdn, [], [1], 1, 1)
        assert_raises(ValueError, upfirdn, [[1]], [1], 1, 1)

    @pytest.mark.parametrize('len_h', [1, 2, 3, 4, 5])
    @pytest.mark.parametrize('len_x', [1, 2, 3, 4, 5])
    def test_singleton(self, len_h, len_x):
        h = np.zeros(len_h)
        h[len_h // 2] = 1.0
        x = np.ones(len_x)
        y = upfirdn(h, x, 1, 1)
        want = np.pad(x, (len_h // 2, (len_h - 1) // 2), 'constant')
        assert_allclose(y, want)

    def test_shift_x(self):
        y = upfirdn([1, 1], [1.0], 1, 1)
        assert_allclose(y, [1, 1])
        y = upfirdn([1, 1], [0.0, 1.0], 1, 1)
        assert_allclose(y, [0, 1, 1])

    @pytest.mark.parametrize('len_h, len_x, up, down, expected', [(2, 2, 5, 2, [1, 0, 0, 0]), (2, 3, 6, 3, [1, 0, 1, 0, 1]), (2, 4, 4, 3, [1, 0, 0, 0, 1]), (3, 2, 6, 2, [1, 0, 0, 1, 0]), (4, 11, 3, 5, [1, 0, 0, 1, 0, 0, 1])])
    def test_length_factors(self, len_h, len_x, up, down, expected):
        h = np.zeros(len_h)
        h[0] = 1.0
        x = np.ones(len_x)
        y = upfirdn(h, x, up, down)
        assert_allclose(y, expected)

    @pytest.mark.parametrize('down, want_len', [(2, 5015), (11, 912), (79, 127)])
    def test_vs_convolve(self, down, want_len):
        random_state = np.random.RandomState(17)
        try_types = (int, np.float32, np.complex64, float, complex)
        size = 10000
        for dtype in try_types:
            x = random_state.randn(size).astype(dtype)
            if dtype in (np.complex64, np.complex128):
                x += 1j * random_state.randn(size)
            h = firwin(31, 1.0 / down, window='hamming')
            yl = upfirdn_naive(x, h, 1, down)
            y = upfirdn(h, x, up=1, down=down)
            assert y.shape == (want_len,)
            assert yl.shape[0] == y.shape[0]
            assert_allclose(yl, y, atol=1e-07, rtol=1e-07)

    @pytest.mark.parametrize('x_dtype', _UPFIRDN_TYPES)
    @pytest.mark.parametrize('h', (1.0, 1j))
    @pytest.mark.parametrize('up, down', [(1, 1), (2, 2), (3, 2), (2, 3)])
    def test_vs_naive_delta(self, x_dtype, h, up, down):
        UpFIRDnCase(up, down, h, x_dtype)()

    @pytest.mark.parametrize('x_dtype', _UPFIRDN_TYPES)
    @pytest.mark.parametrize('h_dtype', _UPFIRDN_TYPES)
    @pytest.mark.parametrize('p_max, q_max', list(product((10, 100), (10, 100))))
    def test_vs_naive(self, x_dtype, h_dtype, p_max, q_max):
        tests = self._random_factors(p_max, q_max, h_dtype, x_dtype)
        for test in tests:
            test()

    def _random_factors(self, p_max, q_max, h_dtype, x_dtype):
        n_rep = 3
        longest_h = 25
        random_state = np.random.RandomState(17)
        tests = []
        for _ in range(n_rep):
            p_add = q_max if p_max > q_max else 1
            q_add = p_max if q_max > p_max else 1
            p = random_state.randint(p_max) + p_add
            q = random_state.randint(q_max) + q_add
            len_h = random_state.randint(longest_h) + 1
            h = np.atleast_1d(random_state.randint(len_h))
            h = h.astype(h_dtype)
            if h_dtype == complex:
                h += 1j * random_state.randint(len_h)
            tests.append(UpFIRDnCase(p, q, h, x_dtype))
        return tests

    @pytest.mark.parametrize('mode', _upfirdn_modes)
    def test_extensions(self, mode):
        """Test vs. manually computed results for modes not in numpy's pad."""
        x = np.array([1, 2, 3, 1], dtype=float)
        npre, npost = (6, 6)
        y = _pad_test(x, npre=npre, npost=npost, mode=mode)
        if mode == 'antisymmetric':
            y_expected = np.asarray([3, 1, -1, -3, -2, -1, 1, 2, 3, 1, -1, -3, -2, -1, 1, 2])
        elif mode == 'antireflect':
            y_expected = np.asarray([1, 2, 3, 1, -1, 0, 1, 2, 3, 1, -1, 0, 1, 2, 3, 1])
        elif mode == 'smooth':
            y_expected = np.asarray([-5, -4, -3, -2, -1, 0, 1, 2, 3, 1, -1, -3, -5, -7, -9, -11])
        elif mode == 'line':
            lin_slope = (x[-1] - x[0]) / (len(x) - 1)
            left = x[0] + np.arange(-npre, 0, 1) * lin_slope
            right = x[-1] + np.arange(1, npost + 1) * lin_slope
            y_expected = np.concatenate((left, x, right))
        else:
            y_expected = np.pad(x, (npre, npost), mode=mode)
        assert_allclose(y, y_expected)

    @pytest.mark.parametrize('size, h_len, mode, dtype', product([8], [4, 5, 26], _upfirdn_modes, [np.float32, np.float64, np.complex64, np.complex128]))
    def test_modes(self, size, h_len, mode, dtype):
        random_state = np.random.RandomState(5)
        x = random_state.randn(size).astype(dtype)
        if dtype in (np.complex64, np.complex128):
            x += 1j * random_state.randn(size)
        h = np.arange(1, 1 + h_len, dtype=x.real.dtype)
        y = upfirdn(h, x, up=1, down=1, mode=mode)
        npad = h_len - 1
        if mode in ['antisymmetric', 'antireflect', 'smooth', 'line']:
            xpad = _pad_test(x, npre=npad, npost=npad, mode=mode)
        else:
            xpad = np.pad(x, npad, mode=mode)
        ypad = upfirdn(h, xpad, up=1, down=1, mode='constant')
        y_expected = ypad[npad:-npad]
        atol = rtol = np.finfo(dtype).eps * 100.0
        assert_allclose(y, y_expected, atol=atol, rtol=rtol)