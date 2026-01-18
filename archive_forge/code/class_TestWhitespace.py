import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
class TestWhitespace:

    def setup_method(self):
        self.A = np.array([['abc ', '123  '], ['789 ', 'xyz ']]).view(np.chararray)
        self.B = np.array([['abc', '123'], ['789', 'xyz']]).view(np.chararray)

    def test1(self):
        assert_(np.all(self.A == self.B))
        assert_(np.all(self.A >= self.B))
        assert_(np.all(self.A <= self.B))
        assert_(not np.any(self.A > self.B))
        assert_(not np.any(self.A < self.B))
        assert_(not np.any(self.A != self.B))