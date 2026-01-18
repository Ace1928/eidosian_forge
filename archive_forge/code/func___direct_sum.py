import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.statespace import structural
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.results import results_structural
def __direct_sum(square_matrices):
    """Compute the matrix direct sum of an iterable of square numpy 2-d arrays
    """
    new_shape = np.sum([m.shape for m in square_matrices], axis=0)
    new_array = np.zeros(new_shape)
    offset = 0
    for m in square_matrices:
        rows, cols = m.shape
        assert rows == cols
        new_array[offset:offset + rows, offset:offset + rows] = m
        offset += rows
    return new_array