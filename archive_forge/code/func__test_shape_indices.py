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
def _test_shape_indices(model):
    for i in range(model.n_clusters):
        m, n = model.get_shape(i)
        i_ind, j_ind = model.get_indices(i)
        assert len(i_ind) == m
        assert len(j_ind) == n