import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
class TestComparisonsMixed2(TestComparisons):
    """Ticket #1276"""

    def setup_method(self):
        TestComparisons.setup_method(self)
        self.A = np.array([['abc', '123'], ['789', 'xyz']], np.str_).view(np.chararray)