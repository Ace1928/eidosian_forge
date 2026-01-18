from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
class _TestIDCTBase:

    def setup_method(self):
        self.rdt = None
        self.dec = 14
        self.type = None

    def test_definition(self):
        for i in FFTWDATA_SIZES:
            xr, yr, dt = fftw_dct_ref(self.type, i, self.rdt)
            x = idct(yr, type=self.type)
            if self.type == 1:
                x /= 2 * (i - 1)
            else:
                x /= 2 * i
            assert_equal(x.dtype, dt)
            assert_array_almost_equal(x / np.max(x), xr / np.max(x), decimal=self.dec, err_msg='Size %d failed' % i)