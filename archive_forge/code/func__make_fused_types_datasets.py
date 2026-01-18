from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris
from sklearn.utils._seq_dataset import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSR_CONTAINERS
def _make_fused_types_datasets():
    all_datasets = _make_dense_datasets() + _make_sparse_datasets()
    return (all_datasets[idx:idx + 2] for idx in range(0, len(all_datasets), 2))