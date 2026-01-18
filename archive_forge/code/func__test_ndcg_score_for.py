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
def _test_ndcg_score_for(y_true, y_score):
    ideal = _ndcg_sample_scores(y_true, y_true)
    score = _ndcg_sample_scores(y_true, y_score)
    assert (score <= ideal).all()
    all_zero = (y_true == 0).all(axis=1)
    assert ideal[~all_zero] == pytest.approx(np.ones((~all_zero).sum()))
    assert ideal[all_zero] == pytest.approx(np.zeros(all_zero.sum()))
    assert score[~all_zero] == pytest.approx(_dcg_sample_scores(y_true, y_score)[~all_zero] / _dcg_sample_scores(y_true, y_true)[~all_zero])
    assert score[all_zero] == pytest.approx(np.zeros(all_zero.sum()))
    assert ideal.shape == (y_true.shape[0],)
    assert score.shape == (y_true.shape[0],)