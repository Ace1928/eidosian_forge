import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
class TestClog:

    def test_simple(self):
        x = np.array([1 + 0j, 1 + 2j])
        y_r = np.log(np.abs(x)) + 1j * np.angle(x)
        y = np.log(x)
        assert_almost_equal(y, y_r)

    @platform_skip
    @pytest.mark.skipif(platform.machine() == 'armv5tel', reason='See gh-413.')
    def test_special_values(self):
        xl = []
        yl = []
        with np.errstate(divide='raise'):
            x = np.array([np.NZERO], dtype=complex)
            y = complex(-np.inf, np.pi)
            assert_raises(FloatingPointError, np.log, x)
        with np.errstate(divide='ignore'):
            assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        with np.errstate(divide='raise'):
            x = np.array([0], dtype=complex)
            y = complex(-np.inf, 0)
            assert_raises(FloatingPointError, np.log, x)
        with np.errstate(divide='ignore'):
            assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(1, np.inf)], dtype=complex)
        y = complex(np.inf, 0.5 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(-1, np.inf)], dtype=complex)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        with np.errstate(invalid='raise'):
            x = np.array([complex(1.0, np.nan)], dtype=complex)
            y = complex(np.nan, np.nan)
        with np.errstate(invalid='ignore'):
            assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        with np.errstate(invalid='raise'):
            x = np.array([np.inf + 1j * np.nan], dtype=complex)
        with np.errstate(invalid='ignore'):
            assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([-np.inf + 1j], dtype=complex)
        y = complex(np.inf, np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([np.inf + 1j], dtype=complex)
        y = complex(np.inf, 0)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(-np.inf, np.inf)], dtype=complex)
        y = complex(np.inf, 0.75 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(np.inf, np.inf)], dtype=complex)
        y = complex(np.inf, 0.25 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(np.inf, np.nan)], dtype=complex)
        y = complex(np.inf, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(-np.inf, np.nan)], dtype=complex)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(np.nan, 1)], dtype=complex)
        y = complex(np.nan, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(np.nan, np.inf)], dtype=complex)
        y = complex(np.inf, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(np.nan, np.nan)], dtype=complex)
        y = complex(np.nan, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        xa = np.array(xl, dtype=complex)
        ya = np.array(yl, dtype=complex)
        with np.errstate(divide='ignore'):
            for i in range(len(xa)):
                assert_almost_equal(np.log(xa[i].conj()), ya[i].conj())