import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
class MyArray2:

    def __init__(self, data):
        self.data = data

    def __array__(self):
        return self.data