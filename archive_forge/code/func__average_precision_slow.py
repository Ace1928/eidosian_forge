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
def _average_precision_slow(y_true, y_score):
    """A second alternative implementation of average precision that closely
    follows the Wikipedia article's definition (see References). This should
    give identical results as `average_precision_score` for all inputs.

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
       <https://en.wikipedia.org/wiki/Average_precision>`_
    """
    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    precision = list(reversed(precision))
    recall = list(reversed(recall))
    average_precision = 0
    for i in range(1, len(precision)):
        average_precision += precision[i] * (recall[i] - recall[i - 1])
    return average_precision