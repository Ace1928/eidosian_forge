import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
class SimpleSubClass(VerySimpleSubClass):

    def __new__(cls, *args, **kwargs):
        self = np.array(*args, subok=True, **kwargs).view(cls)
        self.info = 'simple'
        return self

    def __array_finalize__(self, obj):
        self.info = getattr(obj, 'info', '') + ' finalized'