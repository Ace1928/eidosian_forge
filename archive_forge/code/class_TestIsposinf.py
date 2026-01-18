import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestIsposinf:

    def test_generic(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = isposinf(np.array((-1.0, 0, 1)) / 0.0)
        assert_(vals[0] == 0)
        assert_(vals[1] == 0)
        assert_(vals[2] == 1)