import numpy as np
import pytest
from sklearn.utils._cython_blas import (
from sklearn.utils._testing import assert_allclose
def _numpy_to_cython(dtype):
    cython = pytest.importorskip('cython')
    if dtype == np.float32:
        return cython.float
    elif dtype == np.float64:
        return cython.double