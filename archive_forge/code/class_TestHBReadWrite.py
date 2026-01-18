from io import StringIO
import tempfile
import numpy as np
from numpy.testing import assert_equal, \
from scipy.sparse import coo_matrix, csc_matrix, rand
from scipy.io import hb_read, hb_write
class TestHBReadWrite:

    def check_save_load(self, value):
        with tempfile.NamedTemporaryFile(mode='w+t') as file:
            hb_write(file, value)
            file.file.seek(0)
            value_loaded = hb_read(file)
        assert_csc_almost_equal(value, value_loaded)

    def test_simple(self):
        random_matrix = rand(10, 100, 0.1)
        for matrix_format in ('coo', 'csc', 'csr', 'bsr', 'dia', 'dok', 'lil'):
            matrix = random_matrix.asformat(matrix_format, copy=False)
            self.check_save_load(matrix)