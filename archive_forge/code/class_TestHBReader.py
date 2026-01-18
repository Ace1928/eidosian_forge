from io import StringIO
import tempfile
import numpy as np
from numpy.testing import assert_equal, \
from scipy.sparse import coo_matrix, csc_matrix, rand
from scipy.io import hb_read, hb_write
class TestHBReader:

    def test_simple(self):
        m = hb_read(StringIO(SIMPLE))
        assert_csc_almost_equal(m, SIMPLE_MATRIX)