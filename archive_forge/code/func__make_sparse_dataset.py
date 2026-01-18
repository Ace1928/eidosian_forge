from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris
from sklearn.utils._seq_dataset import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSR_CONTAINERS
def _make_sparse_dataset(csr_container, float_dtype):
    if float_dtype == np.float32:
        X, y, sample_weight, csr_dataset = (X32, y32, sample_weight32, CSRDataset32)
    else:
        X, y, sample_weight, csr_dataset = (X64, y64, sample_weight64, CSRDataset64)
    X = csr_container(X)
    return csr_dataset(X.data, X.indptr, X.indices, y, sample_weight, seed=42)