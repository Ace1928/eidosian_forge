import pytest
from numpy import (
from numpy.testing import (
class PhysicalQuantity(float):

    def __new__(cls, value):
        return float.__new__(cls, value)

    def __add__(self, x):
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(x) + float(self))
    __radd__ = __add__

    def __sub__(self, x):
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(self) - float(x))

    def __rsub__(self, x):
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(x) - float(self))

    def __mul__(self, x):
        return PhysicalQuantity(float(x) * float(self))
    __rmul__ = __mul__

    def __div__(self, x):
        return PhysicalQuantity(float(self) / float(x))

    def __rdiv__(self, x):
        return PhysicalQuantity(float(x) / float(self))