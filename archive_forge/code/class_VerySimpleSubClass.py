import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
class VerySimpleSubClass(np.ndarray):

    def __new__(cls, *args, **kwargs):
        return np.array(*args, subok=True, **kwargs).view(cls)