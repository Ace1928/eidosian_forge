import re
import sys
import warnings
from io import StringIO
import numpy as np
import pytest
from scipy import linalg
from sklearn.base import clone
from sklearn.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from sklearn.decomposition import _nmf as nmf  # For testing internals
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import (
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def _assert_nmf_no_nan(X, beta_loss):
    W, H, _ = non_negative_factorization(X, init='random', n_components=n_components, solver='mu', beta_loss=beta_loss, random_state=0, max_iter=1000)
    assert not np.any(np.isnan(W))
    assert not np.any(np.isnan(H))