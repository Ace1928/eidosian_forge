import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestRealIfClose:

    def test_basic(self):
        a = np.random.rand(10)
        b = real_if_close(a + 1e-15j)
        assert_all(isrealobj(b))
        assert_array_equal(a, b)
        b = real_if_close(a + 1e-07j)
        assert_all(iscomplexobj(b))
        b = real_if_close(a + 1e-07j, tol=1e-06)
        assert_all(isrealobj(b))