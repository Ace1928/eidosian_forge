import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def _test_args(self, a, b):
    if self.a != a or self.b != b:
        raise ValueError()