from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
class _TestDCTBase:

    def setup_method(self):
        self.rdt = None
        self.dec = 14
        self.type = None

    def test_definition(self):
        for i in FFTWDATA_SIZES:
            x, yr, dt = fftw_dct_ref(self.type, i, self.rdt)
            y = dct(x, type=self.type)
            assert_equal(y.dtype, dt)
            assert_array_almost_equal(y / np.max(y), yr / np.max(y), decimal=self.dec, err_msg='Size %d failed' % i)

    def test_axis(self):
        nt = 2
        for i in [7, 8, 9, 16, 32, 64]:
            x = np.random.randn(nt, i)
            y = dct(x, type=self.type)
            for j in range(nt):
                assert_array_almost_equal(y[j], dct(x[j], type=self.type), decimal=self.dec)
            x = x.T
            y = dct(x, axis=0, type=self.type)
            for j in range(nt):
                assert_array_almost_equal(y[:, j], dct(x[:, j], type=self.type), decimal=self.dec)