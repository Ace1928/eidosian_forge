import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
class TestComparisonsMixed1(TestComparisons):
    """Ticket #1276"""

    def setup_method(self):
        TestComparisons.setup_method(self)
        self.B = np.array([['efg', '123  '], ['051', 'tuv']], np.str_).view(np.chararray)