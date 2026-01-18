from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris
from sklearn.utils._seq_dataset import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSR_CONTAINERS
def _make_sparse_datasets():
    return [_make_sparse_dataset(csr_container, float_dtype) for csr_container, float_dtype in product(CSR_CONTAINERS, floating)]