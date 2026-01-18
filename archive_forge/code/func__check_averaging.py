from functools import partial
from inspect import signature
from itertools import chain, permutations, product
import numpy as np
import pytest
from sklearn._config import config_context
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import (
from sklearn.metrics._base import _average_binary_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_random_state
@ignore_warnings
def _check_averaging(metric, y_true, y_pred, y_true_binarize, y_pred_binarize, is_multilabel):
    n_samples, n_classes = y_true_binarize.shape
    label_measure = metric(y_true, y_pred, average=None)
    assert_allclose(label_measure, [metric(y_true_binarize[:, i], y_pred_binarize[:, i]) for i in range(n_classes)])
    micro_measure = metric(y_true, y_pred, average='micro')
    assert_allclose(micro_measure, metric(y_true_binarize.ravel(), y_pred_binarize.ravel()))
    macro_measure = metric(y_true, y_pred, average='macro')
    assert_allclose(macro_measure, np.mean(label_measure))
    weights = np.sum(y_true_binarize, axis=0, dtype=int)
    if np.sum(weights) != 0:
        weighted_measure = metric(y_true, y_pred, average='weighted')
        assert_allclose(weighted_measure, np.average(label_measure, weights=weights))
    else:
        weighted_measure = metric(y_true, y_pred, average='weighted')
        assert_allclose(weighted_measure, 0)
    if is_multilabel:
        sample_measure = metric(y_true, y_pred, average='samples')
        assert_allclose(sample_measure, np.mean([metric(y_true_binarize[i], y_pred_binarize[i]) for i in range(n_samples)]))
    with pytest.raises(ValueError):
        metric(y_true, y_pred, average='unknown')
    with pytest.raises(ValueError):
        metric(y_true, y_pred, average='garbage')