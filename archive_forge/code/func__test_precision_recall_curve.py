import re
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.metrics._ranking import _dcg_sample_scores, _ndcg_sample_scores
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import (
def _test_precision_recall_curve(y_true, y_score, drop):
    p, r, thresholds = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
    precision_recall_auc = _average_precision_slow(y_true, y_score)
    assert_array_almost_equal(precision_recall_auc, 0.859, 3)
    assert_array_almost_equal(precision_recall_auc, average_precision_score(y_true, y_score))
    assert_almost_equal(_average_precision(y_true, y_score), precision_recall_auc, decimal=2)
    assert p.size == r.size
    assert p.size == thresholds.size + 1
    p, r, thresholds = precision_recall_curve(y_true, np.zeros_like(y_score), drop_intermediate=drop)
    assert p.size == r.size
    assert p.size == thresholds.size + 1