import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
class ProbArg:
    """Generate a set of probabilities on [0, 1]."""

    def __init__(self):
        self.a = 0
        self.b = 1

    def values(self, n):
        """Return an array containing approximately n numbers."""
        m = max(1, n // 3)
        v1 = np.logspace(-30, np.log10(0.3), m)
        v2 = np.linspace(0.3, 0.7, m + 1, endpoint=False)[1:]
        v3 = 1 - np.logspace(np.log10(0.3), -15, m)
        v = np.r_[v1, v2, v3]
        return np.unique(v)