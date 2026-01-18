import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
class EndpointFilter:

    def __init__(self, a, b, rtol, atol):
        self.a = a
        self.b = b
        self.rtol = rtol
        self.atol = atol

    def __call__(self, x):
        mask1 = np.abs(x - self.a) < self.rtol * np.abs(self.a) + self.atol
        mask2 = np.abs(x - self.b) < self.rtol * np.abs(self.b) + self.atol
        return np.where(mask1 | mask2, False, True)