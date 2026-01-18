import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
class AssignValues:
    """Check the assignment of unicode arrays with values"""

    def content_check(self, ua, ua_scalar, nbytes):
        assert_(int(ua.dtype.str[2:]) == self.ulen)
        assert_(buffer_length(ua) == nbytes)
        assert_(ua_scalar == self.ucs_value * self.ulen)
        assert_(ua_scalar.encode('utf-8') == (self.ucs_value * self.ulen).encode('utf-8'))
        if self.ucs_value == ucs4_value:
            assert_(buffer_length(ua_scalar) == 2 * 2 * self.ulen)
        else:
            assert_(buffer_length(ua_scalar) == 2 * self.ulen)

    def test_values0D(self):
        ua = np.zeros((), dtype='U%s' % self.ulen)
        ua[()] = self.ucs_value * self.ulen
        self.content_check(ua, ua[()], 4 * self.ulen)

    def test_valuesSD(self):
        ua = np.zeros((2,), dtype='U%s' % self.ulen)
        ua[0] = self.ucs_value * self.ulen
        self.content_check(ua, ua[0], 4 * self.ulen * 2)
        ua[1] = self.ucs_value * self.ulen
        self.content_check(ua, ua[1], 4 * self.ulen * 2)

    def test_valuesMD(self):
        ua = np.zeros((2, 3, 4), dtype='U%s' % self.ulen)
        ua[0, 0, 0] = self.ucs_value * self.ulen
        self.content_check(ua, ua[0, 0, 0], 4 * self.ulen * 2 * 3 * 4)
        ua[-1, -1, -1] = self.ucs_value * self.ulen
        self.content_check(ua, ua[-1, -1, -1], 4 * self.ulen * 2 * 3 * 4)