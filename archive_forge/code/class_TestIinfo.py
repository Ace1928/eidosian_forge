import warnings
import numpy as np
import pytest
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import assert_equal, assert_, assert_raises
from numpy.core.getlimits import _discovered_machar, _float_ma
class TestIinfo:

    def test_basic(self):
        dts = list(zip(['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8'], [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]))
        for dt1, dt2 in dts:
            assert_iinfo_equal(iinfo(dt1), iinfo(dt2))
        assert_raises(ValueError, iinfo, 'f4')

    def test_unsigned_max(self):
        types = np.sctypes['uint']
        for T in types:
            with np.errstate(over='ignore'):
                max_calculated = T(0) - T(1)
            assert_equal(iinfo(T).max, max_calculated)