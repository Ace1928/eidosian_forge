import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def _make_training_data(n_bins=256, constant_hessian=True):
    rng = np.random.RandomState(42)
    n_samples = 10000
    X_binned = rng.randint(0, n_bins - 1, size=(n_samples, 2), dtype=X_BINNED_DTYPE)
    X_binned = np.asfortranarray(X_binned)

    def true_decision_function(input_features):
        """Ground truth decision function

        This is a very simple yet asymmetric decision tree. Therefore the
        grower code should have no trouble recovering the decision function
        from 10000 training samples.
        """
        if input_features[0] <= n_bins // 2:
            return -1
        else:
            return -1 if input_features[1] <= n_bins // 3 else 1
    target = np.array([true_decision_function(x) for x in X_binned], dtype=Y_DTYPE)
    all_gradients = target.astype(G_H_DTYPE)
    shape_hessians = 1 if constant_hessian else all_gradients.shape
    all_hessians = np.ones(shape=shape_hessians, dtype=G_H_DTYPE)
    return (X_binned, all_gradients, all_hessians)