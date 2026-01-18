import numpy as np
import pytest
from sklearn.utils._cython_blas import (
from sklearn.utils._testing import assert_allclose
def _no_op(x):
    return x