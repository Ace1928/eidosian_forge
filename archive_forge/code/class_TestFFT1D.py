import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises as assert_raises
import scipy.fft as fft
from scipy.conftest import (
from scipy._lib._array_api import (
class TestFFT1D:

    @array_api_compatible
    def test_identity(self, xp):
        maxlen = 512
        x = xp.asarray(random(maxlen) + 1j * random(maxlen))
        xr = xp.asarray(random(maxlen))
        for i in range(1, maxlen):
            xp_assert_close(fft.ifft(fft.fft(x[0:i])), x[0:i], rtol=1e-09, atol=0)
            xp_assert_close(fft.irfft(fft.rfft(xr[0:i]), i), xr[0:i], rtol=1e-09, atol=0)

    @array_api_compatible
    def test_fft(self, xp):
        x = random(30) + 1j * random(30)
        expect = xp.asarray(fft1(x))
        x = xp.asarray(x)
        xp_assert_close(fft.fft(x), expect)
        xp_assert_close(fft.fft(x, norm='backward'), expect)
        xp_assert_close(fft.fft(x, norm='ortho'), expect / xp.sqrt(xp.asarray(30, dtype=xp.float64)))
        xp_assert_close(fft.fft(x, norm='forward'), expect / 30)

    @array_api_compatible
    def test_ifft(self, xp):
        x = xp.asarray(random(30) + 1j * random(30))
        xp_assert_close(fft.ifft(fft.fft(x)), x)
        for norm in ['backward', 'ortho', 'forward']:
            xp_assert_close(fft.ifft(fft.fft(x, norm=norm), norm=norm), x)

    @skip_if_array_api_backend('torch')
    @array_api_compatible
    def test_fft2(self, xp):
        x = xp.asarray(random((30, 20)) + 1j * random((30, 20)))
        expect = fft.fft(fft.fft(x, axis=1), axis=0)
        xp_assert_close(fft.fft2(x), expect)
        xp_assert_close(fft.fft2(x, norm='backward'), expect)
        xp_assert_close(fft.fft2(x, norm='ortho'), expect / xp.sqrt(xp.asarray(30 * 20, dtype=xp.float64)))
        xp_assert_close(fft.fft2(x, norm='forward'), expect / (30 * 20))

    @skip_if_array_api_backend('torch')
    @array_api_compatible
    def test_ifft2(self, xp):
        x = xp.asarray(random((30, 20)) + 1j * random((30, 20)))
        expect = fft.ifft(fft.ifft(x, axis=1), axis=0)
        xp_assert_close(fft.ifft2(x), expect)
        xp_assert_close(fft.ifft2(x, norm='backward'), expect)
        xp_assert_close(fft.ifft2(x, norm='ortho'), expect * xp.sqrt(xp.asarray(30 * 20, dtype=xp.float64)))
        xp_assert_close(fft.ifft2(x, norm='forward'), expect * (30 * 20))

    @array_api_compatible
    @skip_if_array_api_backend('torch')
    def test_fftn(self, xp):
        x = xp.asarray(random((30, 20, 10)) + 1j * random((30, 20, 10)))
        expect = fft.fft(fft.fft(fft.fft(x, axis=2), axis=1), axis=0)
        xp_assert_close(fft.fftn(x), expect)
        xp_assert_close(fft.fftn(x, norm='backward'), expect)
        xp_assert_close(fft.fftn(x, norm='ortho'), expect / xp.sqrt(xp.asarray(30 * 20 * 10, dtype=xp.float64)))
        xp_assert_close(fft.fftn(x, norm='forward'), expect / (30 * 20 * 10))

    @array_api_compatible
    @skip_if_array_api_backend('torch')
    def test_ifftn(self, xp):
        x = xp.asarray(random((30, 20, 10)) + 1j * random((30, 20, 10)))
        expect = fft.ifft(fft.ifft(fft.ifft(x, axis=2), axis=1), axis=0)
        xp_assert_close(fft.ifftn(x), expect)
        xp_assert_close(fft.ifftn(x, norm='backward'), expect)
        xp_assert_close(fft.ifftn(x, norm='ortho'), fft.ifftn(x) * xp.sqrt(xp.asarray(30 * 20 * 10, dtype=xp.float64)))
        xp_assert_close(fft.ifftn(x, norm='forward'), expect * (30 * 20 * 10))

    @array_api_compatible
    def test_rfft(self, xp):
        x = xp.asarray(random(29))
        for n in [size(x), 2 * size(x)]:
            for norm in [None, 'backward', 'ortho', 'forward']:
                xp_assert_close(fft.rfft(x, n=n, norm=norm), fft.fft(x, n=n, norm=norm)[:n // 2 + 1])
            xp_assert_close(fft.rfft(x, n=n, norm='ortho'), fft.rfft(x, n=n) / xp.sqrt(xp.asarray(n, dtype=xp.float64)))

    @array_api_compatible
    def test_irfft(self, xp):
        x = xp.asarray(random(30))
        xp_assert_close(fft.irfft(fft.rfft(x)), x)
        for norm in ['backward', 'ortho', 'forward']:
            xp_assert_close(fft.irfft(fft.rfft(x, norm=norm), norm=norm), x)

    @skip_if_array_api_backend('torch')
    @array_api_compatible
    def test_rfft2(self, xp):
        x = xp.asarray(random((30, 20)))
        expect = fft.fft2(x)[:, :11]
        xp_assert_close(fft.rfft2(x), expect)
        xp_assert_close(fft.rfft2(x, norm='backward'), expect)
        xp_assert_close(fft.rfft2(x, norm='ortho'), expect / xp.sqrt(xp.asarray(30 * 20, dtype=xp.float64)))
        xp_assert_close(fft.rfft2(x, norm='forward'), expect / (30 * 20))

    @skip_if_array_api_backend('torch')
    @array_api_compatible
    def test_irfft2(self, xp):
        x = xp.asarray(random((30, 20)))
        xp_assert_close(fft.irfft2(fft.rfft2(x)), x)
        for norm in ['backward', 'ortho', 'forward']:
            xp_assert_close(fft.irfft2(fft.rfft2(x, norm=norm), norm=norm), x)

    @array_api_compatible
    @skip_if_array_api_backend('torch')
    def test_rfftn(self, xp):
        x = xp.asarray(random((30, 20, 10)))
        expect = fft.fftn(x)[:, :, :6]
        xp_assert_close(fft.rfftn(x), expect)
        xp_assert_close(fft.rfftn(x, norm='backward'), expect)
        xp_assert_close(fft.rfftn(x, norm='ortho'), expect / xp.sqrt(xp.asarray(30 * 20 * 10, dtype=xp.float64)))
        xp_assert_close(fft.rfftn(x, norm='forward'), expect / (30 * 20 * 10))

    @array_api_compatible
    @skip_if_array_api_backend('torch')
    def test_irfftn(self, xp):
        x = xp.asarray(random((30, 20, 10)))
        xp_assert_close(fft.irfftn(fft.rfftn(x)), x)
        for norm in ['backward', 'ortho', 'forward']:
            xp_assert_close(fft.irfftn(fft.rfftn(x, norm=norm), norm=norm), x)

    @array_api_compatible
    def test_hfft(self, xp):
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        x = xp.asarray(x)
        x_herm = xp.asarray(x_herm)
        expect = xp.real(fft.fft(x))
        xp_assert_close(fft.hfft(x_herm), expect)
        xp_assert_close(fft.hfft(x_herm, norm='backward'), expect)
        xp_assert_close(fft.hfft(x_herm, norm='ortho'), expect / xp.sqrt(xp.asarray(30, dtype=xp.float64)))
        xp_assert_close(fft.hfft(x_herm, norm='forward'), expect / 30)

    @array_api_compatible
    def test_ihfft(self, xp):
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        x = xp.asarray(x)
        x_herm = xp.asarray(x_herm)
        xp_assert_close(fft.ihfft(fft.hfft(x_herm)), x_herm)
        for norm in ['backward', 'ortho', 'forward']:
            xp_assert_close(fft.ihfft(fft.hfft(x_herm, norm=norm), norm=norm), x_herm)

    @skip_if_array_api_backend('torch')
    @array_api_compatible
    def test_hfft2(self, xp):
        x = xp.asarray(random((30, 20)))
        xp_assert_close(fft.hfft2(fft.ihfft2(x)), x)
        for norm in ['backward', 'ortho', 'forward']:
            xp_assert_close(fft.hfft2(fft.ihfft2(x, norm=norm), norm=norm), x)

    @skip_if_array_api_backend('torch')
    @array_api_compatible
    def test_ihfft2(self, xp):
        x = xp.asarray(random((30, 20)))
        expect = fft.ifft2(x)[:, :11]
        xp_assert_close(fft.ihfft2(x), expect)
        xp_assert_close(fft.ihfft2(x, norm='backward'), expect)
        xp_assert_close(fft.ihfft2(x, norm='ortho'), expect * xp.sqrt(xp.asarray(30 * 20, dtype=xp.float64)))
        xp_assert_close(fft.ihfft2(x, norm='forward'), expect * (30 * 20))

    @skip_if_array_api_backend('torch')
    @array_api_compatible
    def test_hfftn(self, xp):
        x = xp.asarray(random((30, 20, 10)))
        xp_assert_close(fft.hfftn(fft.ihfftn(x)), x)
        for norm in ['backward', 'ortho', 'forward']:
            xp_assert_close(fft.hfftn(fft.ihfftn(x, norm=norm), norm=norm), x)

    @array_api_compatible
    @skip_if_array_api_backend('torch')
    def test_ihfftn(self, xp):
        x = xp.asarray(random((30, 20, 10)))
        expect = fft.ifftn(x)[:, :, :6]
        xp_assert_close(expect, fft.ihfftn(x))
        xp_assert_close(expect, fft.ihfftn(x, norm='backward'))
        xp_assert_close(fft.ihfftn(x, norm='ortho'), expect * xp.sqrt(xp.asarray(30 * 20 * 10, dtype=xp.float64)))
        xp_assert_close(fft.ihfftn(x, norm='forward'), expect * (30 * 20 * 10))

    def _check_axes(self, op, xp):
        x = xp.asarray(random((30, 20, 10)))
        axes = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        xp_test = array_namespace(x)
        for a in axes:
            op_tr = op(xp_test.permute_dims(x, axes=a))
            tr_op = xp_test.permute_dims(op(x, axes=a), axes=a)
            xp_assert_close(op_tr, tr_op)

    @skip_if_array_api_backend('torch')
    @array_api_compatible
    @pytest.mark.parametrize('op', [fft.fftn, fft.ifftn, fft.rfftn, fft.irfftn])
    def test_axes_standard(self, op, xp):
        self._check_axes(op, xp)

    @skip_if_array_api_backend('torch')
    @array_api_compatible
    @pytest.mark.parametrize('op', [fft.hfftn, fft.ihfftn])
    def test_axes_non_standard(self, op, xp):
        self._check_axes(op, xp)

    @skip_if_array_api_backend('torch')
    @array_api_compatible
    @pytest.mark.parametrize('op', [fft.fftn, fft.ifftn, fft.rfftn, fft.irfftn])
    def test_axes_subset_with_shape_standard(self, op, xp):
        x = xp.asarray(random((16, 8, 4)))
        axes = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        xp_test = array_namespace(x)
        for a in axes:
            shape = tuple([2 * x.shape[ax] if ax in a[:2] else x.shape[ax] for ax in range(x.ndim)])
            op_tr = op(xp_test.permute_dims(x, axes=a), s=shape[:2], axes=(0, 1))
            tr_op = xp_test.permute_dims(op(x, s=shape[:2], axes=a[:2]), axes=a)
            xp_assert_close(op_tr, tr_op)

    @skip_if_array_api_backend('torch')
    @array_api_compatible
    @pytest.mark.parametrize('op', [fft.fft2, fft.ifft2, fft.rfft2, fft.irfft2, fft.hfft2, fft.ihfft2, fft.hfftn, fft.ihfftn])
    def test_axes_subset_with_shape_non_standard(self, op, xp):
        x = xp.asarray(random((16, 8, 4)))
        axes = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        xp_test = array_namespace(x)
        for a in axes:
            shape = tuple([2 * x.shape[ax] if ax in a[:2] else x.shape[ax] for ax in range(x.ndim)])
            op_tr = op(xp_test.permute_dims(x, axes=a), s=shape[:2], axes=(0, 1))
            tr_op = xp_test.permute_dims(op(x, s=shape[:2], axes=a[:2]), axes=a)
            xp_assert_close(op_tr, tr_op)

    @array_api_compatible
    def test_all_1d_norm_preserving(self, xp):
        x = xp.asarray(random(30))
        xp_test = array_namespace(x)
        x_norm = xp_test.linalg.vector_norm(x)
        n = size(x) * 2
        func_pairs = [(fft.fft, fft.ifft), (fft.rfft, fft.irfft), (fft.ihfft, fft.hfft)]
        for forw, back in func_pairs:
            for n in [size(x), 2 * size(x)]:
                for norm in ['backward', 'ortho', 'forward']:
                    tmp = forw(x, n=n, norm=norm)
                    tmp = back(tmp, n=n, norm=norm)
                    xp_assert_close(xp_test.linalg.vector_norm(tmp), x_norm)

    @pytest.mark.parametrize('dtype', [np.float16, np.longdouble])
    def test_dtypes_nonstandard(self, dtype):
        x = random(30).astype(dtype)
        out_dtypes = {np.float16: np.complex64, np.longdouble: np.clongdouble}
        x_complex = x.astype(out_dtypes[dtype])
        res_fft = fft.ifft(fft.fft(x))
        res_rfft = fft.irfft(fft.rfft(x))
        res_hfft = fft.hfft(fft.ihfft(x), x.shape[0])
        assert_array_almost_equal(res_fft, x_complex)
        assert_array_almost_equal(res_rfft, x)
        assert_array_almost_equal(res_hfft, x)
        assert res_fft.dtype == x_complex.dtype
        assert res_rfft.dtype == np.result_type(np.float32, x.dtype)
        assert res_hfft.dtype == np.result_type(np.float32, x.dtype)

    @array_api_compatible
    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    def test_dtypes(self, dtype, xp):
        x = xp.asarray(random(30), dtype=getattr(xp, dtype))
        out_dtypes = {'float32': xp.complex64, 'float64': xp.complex128}
        x_complex = xp.asarray(x, dtype=out_dtypes[dtype])
        res_fft = fft.ifft(fft.fft(x))
        res_rfft = fft.irfft(fft.rfft(x))
        res_hfft = fft.hfft(fft.ihfft(x), x.shape[0])
        rtol = {'float32': 0.00012, 'float64': 1e-08}[dtype]
        xp_assert_close(res_fft, x_complex, rtol=rtol, atol=0)
        xp_assert_close(res_rfft, x, rtol=rtol, atol=0)
        xp_assert_close(res_hfft, x, rtol=rtol, atol=0)