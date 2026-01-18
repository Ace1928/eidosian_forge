import numpy as np
from skimage.measure import points_in_poly, grid_points_in_poly
from skimage._shared.testing import assert_array_equal
class TestNpnpoly:

    def test_square(self):
        v = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert points_in_poly([[0.5, 0.5]], v)[0]
        assert not points_in_poly([[-0.1, 0.1]], v)[0]

    def test_triangle(self):
        v = np.array([[0, 0], [1, 0], [0.5, 0.75]])
        assert points_in_poly([[0.5, 0.7]], v)[0]
        assert not points_in_poly([[0.5, 0.76]], v)[0]
        assert not points_in_poly([[0.7, 0.5]], v)[0]

    def test_type(self):
        assert points_in_poly([[0, 0]], [[0, 0]]).dtype == bool