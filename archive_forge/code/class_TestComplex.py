from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
class TestComplex:

    def test_dct_complex64(self):
        y = dct(1j * np.arange(5, dtype=np.complex64))
        x = 1j * dct(np.arange(5))
        assert_array_almost_equal(x, y)

    def test_dct_complex(self):
        y = dct(np.arange(5) * 1j)
        x = 1j * dct(np.arange(5))
        assert_array_almost_equal(x, y)

    def test_idct_complex(self):
        y = idct(np.arange(5) * 1j)
        x = 1j * idct(np.arange(5))
        assert_array_almost_equal(x, y)

    def test_dst_complex64(self):
        y = dst(np.arange(5, dtype=np.complex64) * 1j)
        x = 1j * dst(np.arange(5))
        assert_array_almost_equal(x, y)

    def test_dst_complex(self):
        y = dst(np.arange(5) * 1j)
        x = 1j * dst(np.arange(5))
        assert_array_almost_equal(x, y)

    def test_idst_complex(self):
        y = idst(np.arange(5) * 1j)
        x = 1j * idst(np.arange(5))
        assert_array_almost_equal(x, y)