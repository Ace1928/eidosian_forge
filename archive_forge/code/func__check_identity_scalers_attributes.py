import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
def _check_identity_scalers_attributes(scaler_1, scaler_2):
    assert scaler_1.mean_ is scaler_2.mean_ is None
    assert scaler_1.var_ is scaler_2.var_ is None
    assert scaler_1.scale_ is scaler_2.scale_ is None
    assert scaler_1.n_samples_seen_ == scaler_2.n_samples_seen_