import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class TestSctypeDict:

    def test_longdouble(self):
        assert_(np.sctypeDict['f8'] is not np.longdouble)
        assert_(np.sctypeDict['c16'] is not np.clongdouble)

    def test_ulong(self):
        assert_(np.sctypeDict['ulong'] is np.uint)
        with pytest.warns(FutureWarning):
            assert not hasattr(np, 'ulong')