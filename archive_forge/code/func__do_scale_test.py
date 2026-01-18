import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from sklearn.cluster._bicluster import (
from sklearn.datasets import make_biclusters, make_checkerboard
from sklearn.metrics import consensus_score, v_measure_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def _do_scale_test(scaled):
    """Check that rows sum to one constant, and columns to another."""
    row_sum = scaled.sum(axis=1)
    col_sum = scaled.sum(axis=0)
    if issparse(scaled):
        row_sum = np.asarray(row_sum).squeeze()
        col_sum = np.asarray(col_sum).squeeze()
    assert_array_almost_equal(row_sum, np.tile(row_sum.mean(), 100), decimal=1)
    assert_array_almost_equal(col_sum, np.tile(col_sum.mean(), 100), decimal=1)