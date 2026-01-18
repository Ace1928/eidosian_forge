import itertools
import re
import warnings
from functools import partial
import numpy as np
import pytest
import threadpoolctl
from scipy.spatial.distance import cdist
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics._pairwise_distances_reduction import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def _non_trivial_radius(*, X=None, Y=None, metric=None, precomputed_dists=None, expected_n_neighbors=10, n_subsampled_queries=10, **metric_kwargs):
    assert precomputed_dists is not None or metric is not None, 'Either metric or precomputed_dists must be provided.'
    if precomputed_dists is None:
        assert X is not None
        assert Y is not None
        sampled_dists = pairwise_distances(X, Y, metric=metric, **metric_kwargs)
    else:
        sampled_dists = precomputed_dists[:n_subsampled_queries].copy()
    sampled_dists.sort(axis=1)
    return sampled_dists[:, expected_n_neighbors].mean()