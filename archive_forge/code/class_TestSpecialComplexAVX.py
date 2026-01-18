import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
class TestSpecialComplexAVX:

    @pytest.mark.parametrize('stride', [-4, -2, -1, 1, 2, 4])
    @pytest.mark.parametrize('astype', [np.complex64, np.complex128])
    def test_array(self, stride, astype):
        arr = np.array([complex(np.nan, np.nan), complex(np.nan, np.inf), complex(np.inf, np.nan), complex(np.inf, np.inf), complex(0.0, np.inf), complex(np.inf, 0.0), complex(0.0, 0.0), complex(0.0, np.nan), complex(np.nan, 0.0)], dtype=astype)
        abs_true = np.array([np.nan, np.inf, np.inf, np.inf, np.inf, np.inf, 0.0, np.nan, np.nan], dtype=arr.real.dtype)
        sq_true = np.array([complex(np.nan, np.nan), complex(np.nan, np.nan), complex(np.nan, np.nan), complex(np.nan, np.inf), complex(-np.inf, np.nan), complex(np.inf, np.nan), complex(0.0, 0.0), complex(np.nan, np.nan), complex(np.nan, np.nan)], dtype=astype)
        with np.errstate(invalid='ignore'):
            assert_equal(np.abs(arr[::stride]), abs_true[::stride])
            assert_equal(np.square(arr[::stride]), sq_true[::stride])