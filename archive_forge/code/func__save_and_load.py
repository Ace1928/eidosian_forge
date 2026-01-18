import os
import numpy as np
import tempfile
from pytest import raises as assert_raises
from numpy.testing import assert_equal, assert_
from scipy.sparse import (sparray, csc_matrix, csr_matrix, bsr_matrix, dia_matrix,
def _save_and_load(matrix):
    fd, tmpfile = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    try:
        save_npz(tmpfile, matrix)
        loaded_matrix = load_npz(tmpfile)
    finally:
        os.remove(tmpfile)
    return loaded_matrix