from numpy.testing import (
from numpy import (
import numpy as np
import pytest
class TestEye:

    def test_basic(self):
        assert_equal(eye(4), array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        assert_equal(eye(4, dtype='f'), array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 'f'))
        assert_equal(eye(3) == 1, eye(3, dtype=bool))

    def test_uint64(self):
        assert_equal(eye(np.uint64(2), dtype=int), array([[1, 0], [0, 1]]))
        assert_equal(eye(np.uint64(2), M=np.uint64(4), k=np.uint64(1)), array([[0, 1, 0, 0], [0, 0, 1, 0]]))

    def test_diag(self):
        assert_equal(eye(4, k=1), array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]))
        assert_equal(eye(4, k=-1), array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

    def test_2d(self):
        assert_equal(eye(4, 3), array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        assert_equal(eye(3, 4), array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

    def test_diag2d(self):
        assert_equal(eye(3, 4, k=2), array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]))
        assert_equal(eye(4, 3, k=-2), array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]]))

    def test_eye_bounds(self):
        assert_equal(eye(2, 2, 1), [[0, 1], [0, 0]])
        assert_equal(eye(2, 2, -1), [[0, 0], [1, 0]])
        assert_equal(eye(2, 2, 2), [[0, 0], [0, 0]])
        assert_equal(eye(2, 2, -2), [[0, 0], [0, 0]])
        assert_equal(eye(3, 2, 2), [[0, 0], [0, 0], [0, 0]])
        assert_equal(eye(3, 2, 1), [[0, 1], [0, 0], [0, 0]])
        assert_equal(eye(3, 2, -1), [[0, 0], [1, 0], [0, 1]])
        assert_equal(eye(3, 2, -2), [[0, 0], [0, 0], [1, 0]])
        assert_equal(eye(3, 2, -3), [[0, 0], [0, 0], [0, 0]])

    def test_strings(self):
        assert_equal(eye(2, 2, dtype='S3'), [[b'1', b''], [b'', b'1']])

    def test_bool(self):
        assert_equal(eye(2, 2, dtype=bool), [[True, False], [False, True]])

    def test_order(self):
        mat_c = eye(4, 3, k=-1)
        mat_f = eye(4, 3, k=-1, order='F')
        assert_equal(mat_c, mat_f)
        assert mat_c.flags.c_contiguous
        assert not mat_c.flags.f_contiguous
        assert not mat_f.flags.c_contiguous
        assert mat_f.flags.f_contiguous