from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
class _TestDCTIVBase(_TestDCTBase):

    def test_definition_ortho(self):
        dt = np.result_type(np.float32, self.rdt)
        for xr in X:
            x = np.array(xr, dtype=self.rdt)
            y = dct(x, norm='ortho', type=4)
            y2 = naive_dct4(x, norm='ortho')
            assert_equal(y.dtype, dt)
            assert_array_almost_equal(y / np.max(y), y2 / np.max(y), decimal=self.dec)