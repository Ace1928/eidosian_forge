import pytest
import numpy as np
from numpy.testing import assert_, assert_raises
class TestInherit:

    def test_init(self):
        x = B(1.0)
        assert_(str(x) == '1.0')
        y = C(2.0)
        assert_(str(y) == '2.0')
        z = D(3.0)
        assert_(str(z) == '3.0')

    def test_init2(self):
        x = B0(1.0)
        assert_(str(x) == '1.0')
        y = C0(2.0)
        assert_(str(y) == '2.0')

    def test_gh_15395(self):
        x = B1(1.0)
        assert_(str(x) == '1.0')
        with pytest.raises(TypeError):
            B1(1.0, 2.0)