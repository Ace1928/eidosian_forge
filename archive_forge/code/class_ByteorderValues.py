import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
class ByteorderValues:
    """Check the byteorder of unicode arrays in round-trip conversions"""

    def test_values0D(self):
        ua = np.array(self.ucs_value * self.ulen, dtype='U%s' % self.ulen)
        ua2 = ua.newbyteorder()
        assert_(ua[()] != ua2[()])
        ua3 = ua2.newbyteorder()
        assert_equal(ua, ua3)

    def test_valuesSD(self):
        ua = np.array([self.ucs_value * self.ulen] * 2, dtype='U%s' % self.ulen)
        ua2 = ua.newbyteorder()
        assert_((ua != ua2).all())
        assert_(ua[-1] != ua2[-1])
        ua3 = ua2.newbyteorder()
        assert_equal(ua, ua3)

    def test_valuesMD(self):
        ua = np.array([[[self.ucs_value * self.ulen] * 2] * 3] * 4, dtype='U%s' % self.ulen)
        ua2 = ua.newbyteorder()
        assert_((ua != ua2).all())
        assert_(ua[-1, -1, -1] != ua2[-1, -1, -1])
        ua3 = ua2.newbyteorder()
        assert_equal(ua, ua3)

    def test_values_cast(self):
        test1 = np.array([self.ucs_value * self.ulen] * 2, dtype='U%s' % self.ulen)
        test2 = np.repeat(test1, 2)[::2]
        for ua in (test1, test2):
            ua2 = ua.astype(dtype=ua.dtype.newbyteorder())
            assert_((ua == ua2).all())
            assert_(ua[-1] == ua2[-1])
            ua3 = ua2.astype(dtype=ua.dtype)
            assert_equal(ua, ua3)

    def test_values_updowncast(self):
        test1 = np.array([self.ucs_value * self.ulen] * 2, dtype='U%s' % self.ulen)
        test2 = np.repeat(test1, 2)[::2]
        for ua in (test1, test2):
            longer_type = np.dtype('U%s' % (self.ulen + 1)).newbyteorder()
            ua2 = ua.astype(dtype=longer_type)
            assert_((ua == ua2).all())
            assert_(ua[-1] == ua2[-1])
            ua3 = ua2.astype(dtype=ua.dtype)
            assert_equal(ua, ua3)