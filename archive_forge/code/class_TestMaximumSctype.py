import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class TestMaximumSctype:

    @pytest.mark.parametrize('t', [np.byte, np.short, np.intc, np.int_, np.longlong])
    def test_int(self, t):
        assert_equal(np.maximum_sctype(t), np.sctypes['int'][-1])

    @pytest.mark.parametrize('t', [np.ubyte, np.ushort, np.uintc, np.uint, np.ulonglong])
    def test_uint(self, t):
        assert_equal(np.maximum_sctype(t), np.sctypes['uint'][-1])

    @pytest.mark.parametrize('t', [np.half, np.single, np.double, np.longdouble])
    def test_float(self, t):
        assert_equal(np.maximum_sctype(t), np.sctypes['float'][-1])

    @pytest.mark.parametrize('t', [np.csingle, np.cdouble, np.clongdouble])
    def test_complex(self, t):
        assert_equal(np.maximum_sctype(t), np.sctypes['complex'][-1])

    @pytest.mark.parametrize('t', [np.bool_, np.object_, np.str_, np.bytes_, np.void])
    def test_other(self, t):
        assert_equal(np.maximum_sctype(t), t)