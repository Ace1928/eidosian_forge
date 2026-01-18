import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def _sprandn(m, n, density=0.01, format='coo', dtype=None, random_state=None):
    random_state = check_random_state(random_state)
    data_rvs = random_state.standard_normal
    return construct.random(m, n, density, format, dtype, random_state, data_rvs)