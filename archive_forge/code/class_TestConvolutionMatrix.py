import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
class TestConvolutionMatrix:
    """
    Test convolution_matrix vs. numpy.convolve for various parameters.
    """

    def create_vector(self, n, cpx):
        """Make a complex or real test vector of length n."""
        x = np.linspace(-2.5, 2.2, n)
        if cpx:
            x = x + 1j * np.linspace(-1.5, 3.1, n)
        return x

    def test_bad_n(self):
        with pytest.raises(ValueError, match='n must be a positive integer'):
            convolution_matrix([1, 2, 3], 0)

    def test_bad_first_arg(self):
        with pytest.raises(ValueError, match='one-dimensional'):
            convolution_matrix(1, 4)

    def test_empty_first_arg(self):
        with pytest.raises(ValueError, match='len\\(a\\)'):
            convolution_matrix([], 4)

    def test_bad_mode(self):
        with pytest.raises(ValueError, match='mode.*must be one of'):
            convolution_matrix((1, 1), 4, mode='invalid argument')

    @pytest.mark.parametrize('cpx', [False, True])
    @pytest.mark.parametrize('na', [1, 2, 9])
    @pytest.mark.parametrize('nv', [1, 2, 9])
    @pytest.mark.parametrize('mode', [None, 'full', 'valid', 'same'])
    def test_against_numpy_convolve(self, cpx, na, nv, mode):
        a = self.create_vector(na, cpx)
        v = self.create_vector(nv, cpx)
        if mode is None:
            y1 = np.convolve(v, a)
            A = convolution_matrix(a, nv)
        else:
            y1 = np.convolve(v, a, mode)
            A = convolution_matrix(a, nv, mode)
        y2 = A @ v
        assert_array_almost_equal(y1, y2)