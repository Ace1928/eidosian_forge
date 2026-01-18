import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestIsscalar:

    def test_basic(self):
        assert_(np.isscalar(3))
        assert_(not np.isscalar([3]))
        assert_(not np.isscalar((3,)))
        assert_(np.isscalar(3j))
        assert_(np.isscalar(4.0))