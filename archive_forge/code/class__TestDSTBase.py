from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
class _TestDSTBase:

    def setup_method(self):
        self.rdt = None
        self.dec = None
        self.type = None

    def test_definition(self):
        for i in FFTWDATA_SIZES:
            xr, yr, dt = fftw_dst_ref(self.type, i, self.rdt)
            y = dst(xr, type=self.type)
            assert_equal(y.dtype, dt)
            assert_array_almost_equal(y / np.max(y), yr / np.max(y), decimal=self.dec, err_msg='Size %d failed' % i)