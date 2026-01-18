import warnings
from collections.abc import Sequence
from itertools import chain
import numpy as np
from scipy.sparse import issparse
from ..utils._array_api import get_namespace
from ..utils.fixes import VisibleDeprecationWarning
from .validation import _assert_all_finite, check_array
def _ovr_decision_function(predictions, confidences, n_classes):
    """Compute a continuous, tie-breaking OvR decision function from OvO.

    It is important to include a continuous value, not only votes,
    to make computing AUC or calibration meaningful.

    Parameters
    ----------
    predictions : array-like of shape (n_samples, n_classifiers)
        Predicted classes for each binary classifier.

    confidences : array-like of shape (n_samples, n_classifiers)
        Decision functions or predicted probabilities for positive class
        for each binary classifier.

    n_classes : int
        Number of classes. n_classifiers must be
        ``n_classes * (n_classes - 1 ) / 2``.
    """
    n_samples = predictions.shape[0]
    votes = np.zeros((n_samples, n_classes))
    sum_of_confidences = np.zeros((n_samples, n_classes))
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            sum_of_confidences[:, i] -= confidences[:, k]
            sum_of_confidences[:, j] += confidences[:, k]
            votes[predictions[:, k] == 0, i] += 1
            votes[predictions[:, k] == 1, j] += 1
            k += 1
    transformed_confidences = sum_of_confidences / (3 * (np.abs(sum_of_confidences) + 1))
    return votes + transformed_confidences