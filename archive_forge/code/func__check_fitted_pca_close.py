import re
import warnings
import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_array_equal
from sklearn import config_context, datasets
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import _assess_dimension, _infer_dimension
from sklearn.utils._array_api import (
from sklearn.utils._array_api import device as array_device
from sklearn.utils._testing import _array_api_for_tests, assert_allclose
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def _check_fitted_pca_close(pca1, pca2, rtol):
    assert_allclose(pca1.components_, pca2.components_, rtol=rtol)
    assert_allclose(pca1.explained_variance_, pca2.explained_variance_, rtol=rtol)
    assert_allclose(pca1.singular_values_, pca2.singular_values_, rtol=rtol)
    assert_allclose(pca1.mean_, pca2.mean_, rtol=rtol)
    assert_allclose(pca1.n_components_, pca2.n_components_, rtol=rtol)
    assert_allclose(pca1.n_samples_, pca2.n_samples_, rtol=rtol)
    assert_allclose(pca1.noise_variance_, pca2.noise_variance_, rtol=rtol)
    assert_allclose(pca1.n_features_in_, pca2.n_features_in_, rtol=rtol)