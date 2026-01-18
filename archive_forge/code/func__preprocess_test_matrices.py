from . import plot
from . import select
from . import utils
from ._lazyload import matplotlib
from scipy import sparse
from scipy import stats
from sklearn import metrics
from sklearn import neighbors
import joblib
import numbers
import numpy as np
import pandas as pd
import warnings
def _preprocess_test_matrices(X, Y):
    X = utils.to_array_or_spmatrix(X)
    Y = utils.to_array_or_spmatrix(Y)
    if not X.shape[1] == Y.shape[1]:
        raise ValueError('Expected X and Y to have the same number of columns. Got shapes {}, {}'.format(X.shape, Y.shape))
    return (X, Y)