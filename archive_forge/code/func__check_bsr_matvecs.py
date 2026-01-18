import sys
import os
import gc
import threading
import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from scipy.sparse import (_sparsetools, coo_matrix, csr_matrix, csc_matrix,
from scipy.sparse._sputils import supported_dtypes
from scipy._lib._testutils import check_free_memory
import pytest
from pytest import raises as assert_raises
def _check_bsr_matvecs(self, m):
    m = m()
    n = self.n
    r = m.dot(np.ones((n, 2), dtype=np.int8))
    assert_equal(r[0, 0], int_to_int8(n))