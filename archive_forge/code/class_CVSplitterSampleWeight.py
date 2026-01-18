import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
class CVSplitterSampleWeight(CVSplitter):

    def split(self, X, y=None, groups=None, sample_weight=None):
        split_index = len(X) // 2
        train_indices = list(range(0, split_index))
        test_indices = list(range(split_index, len(X)))
        yield (test_indices, train_indices)
        yield (train_indices, test_indices)