import copy
import itertools
import re
import sys
import warnings
from io import StringIO
from unittest.mock import Mock
import numpy as np
import pytest
from scipy import linalg, stats
import sklearn
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_spd_matrix
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def _naive_lmvnpdf_diag(X, means, covars):
    resp = np.empty((len(X), len(means)))
    stds = np.sqrt(covars)
    for i, (mean, std) in enumerate(zip(means, stds)):
        resp[:, i] = stats.norm.logpdf(X, mean, std).sum(axis=1)
    return resp