import os
import pytest
import numpy as np
from . import util
class TestIntentInOut(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'regression', 'inout.f90')]

    @pytest.mark.slow
    def test_inout(self):
        x = np.arange(6, dtype=np.float32)[::2]
        pytest.raises(ValueError, self.module.foo, x)
        x = np.arange(3, dtype=np.float32)
        self.module.foo(x)
        assert np.allclose(x, [3, 1, 2])