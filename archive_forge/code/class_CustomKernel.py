import re
import sys
import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import (
class CustomKernel(C):
    """
    A custom kernel that has a diag method that returns the first column of the
    input matrix X. This is a helper for the test to check that the input
    matrix X is not mutated.
    """

    def diag(self, X):
        return X[:, 0]