import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
class TestComplexAbsoluteAVX:

    @pytest.mark.parametrize('arraysize', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 18, 19])
    @pytest.mark.parametrize('stride', [-4, -3, -2, -1, 1, 2, 3, 4])
    @pytest.mark.parametrize('astype', [np.complex64, np.complex128])
    def test_array(self, arraysize, stride, astype):
        arr = np.ones(arraysize, dtype=astype)
        abs_true = np.ones(arraysize, dtype=arr.real.dtype)
        assert_equal(np.abs(arr[::stride]), abs_true[::stride])