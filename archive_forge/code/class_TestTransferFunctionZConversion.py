import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
class TestTransferFunctionZConversion:
    """Test private conversions between 'z' and 'z**-1' polynomials."""

    def test_full(self):
        num = [2, 3, 4]
        den = [5, 6, 7]
        num2, den2 = TransferFunction._z_to_zinv(num, den)
        assert_equal(num, num2)
        assert_equal(den, den2)
        num2, den2 = TransferFunction._zinv_to_z(num, den)
        assert_equal(num, num2)
        assert_equal(den, den2)

    def test_numerator(self):
        num = [2, 3]
        den = [5, 6, 7]
        num2, den2 = TransferFunction._z_to_zinv(num, den)
        assert_equal([0, 2, 3], num2)
        assert_equal(den, den2)
        num2, den2 = TransferFunction._zinv_to_z(num, den)
        assert_equal([2, 3, 0], num2)
        assert_equal(den, den2)

    def test_denominator(self):
        num = [2, 3, 4]
        den = [5, 6]
        num2, den2 = TransferFunction._z_to_zinv(num, den)
        assert_equal(num, num2)
        assert_equal([0, 5, 6], den2)
        num2, den2 = TransferFunction._zinv_to_z(num, den)
        assert_equal(num, num2)
        assert_equal([5, 6, 0], den2)