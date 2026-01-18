import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.cross_decomposition._pls import (
from sklearn.datasets import load_linnerud, make_regression
from sklearn.ensemble import VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
def _generate_test_scale_and_stability_datasets():
    """Generate dataset for test_scale_and_stability"""
    rng = np.random.RandomState(0)
    n_samples = 1000
    n_targets = 5
    n_features = 10
    Q = rng.randn(n_targets, n_features)
    Y = rng.randn(n_samples, n_targets)
    X = np.dot(Y, Q) + 2 * rng.randn(n_samples, n_features) + 1
    X *= 1000
    yield (X, Y)
    X, Y = load_linnerud(return_X_y=True)
    X[:, -1] = 1.0
    yield (X, Y)
    X = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0], [3.0, 5.0, 4.0]])
    Y = np.array([[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])
    yield (X, Y)
    seeds = [530, 741]
    for seed in seeds:
        rng = np.random.RandomState(seed)
        X = rng.randn(4, 3)
        Y = rng.randn(4, 2)
        yield (X, Y)