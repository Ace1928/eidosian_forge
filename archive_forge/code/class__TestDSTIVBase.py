from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
class _TestDSTIVBase(_TestDSTBase):

    def test_definition_ortho(self):
        dt = np.result_type(np.float32, self.rdt)
        for xr in X:
            x = np.array(xr, dtype=self.rdt)
            y = dst(x, norm='ortho', type=4)
            y2 = naive_dst4(x, norm='ortho')
            assert_equal(y.dtype, dt)
            assert_array_almost_equal(y, y2, decimal=self.dec)