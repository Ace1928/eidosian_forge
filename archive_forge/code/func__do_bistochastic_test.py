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
def _do_bistochastic_test(scaled):
    """Check that rows and columns sum to the same constant."""
    _do_scale_test(scaled)
    assert_almost_equal(scaled.sum(axis=0).mean(), scaled.sum(axis=1).mean(), decimal=1)