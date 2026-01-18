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
def _partial_roc(y_true, y_predict, max_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_predict)
    new_fpr = fpr[fpr <= max_fpr]
    new_fpr = np.append(new_fpr, max_fpr)
    new_tpr = tpr[fpr <= max_fpr]
    idx_out = np.argmax(fpr > max_fpr)
    idx_in = idx_out - 1
    x_interp = [fpr[idx_in], fpr[idx_out]]
    y_interp = [tpr[idx_in], tpr[idx_out]]
    new_tpr = np.append(new_tpr, np.interp(max_fpr, x_interp, y_interp))
    return (new_fpr, new_tpr)