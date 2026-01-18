import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
class ArrayNoInheritance:
    """Quantity-like class that does not inherit from ndarray"""

    def __init__(self, data, units):
        self.magnitude = data
        self.units = units

    def __getattr__(self, attr):
        return getattr(self.magnitude, attr)