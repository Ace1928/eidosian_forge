import os
import pytest
import numpy as np
from . import util
class TestNegativeBounds(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'negative_bounds', 'issue_20853.f90')]

    @pytest.mark.slow
    def test_negbound(self):
        xvec = np.arange(12)
        xlow = -6
        xhigh = 4

        def ubound(xl, xh):
            return xh - xl + 1
        rval = self.module.foo(is_=xlow, ie_=xhigh, arr=xvec[:ubound(xlow, xhigh)])
        expval = np.arange(11, dtype=np.float32)
        assert np.allclose(rval, expval)