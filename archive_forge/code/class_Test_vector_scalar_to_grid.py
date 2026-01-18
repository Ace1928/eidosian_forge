import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.vector_transform as vec_trans
class Test_vector_scalar_to_grid:

    @classmethod
    def setup_class(cls):
        cls.x, cls.y = _sample_plate_carree_coordinates()
        cls.u, cls.v = _sample_plate_carree_vector_field()
        cls.s = _sample_plate_carree_scalar_field()

    def test_no_transform(self):
        expected_x_grid = np.array([[-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0]])
        expected_y_grid = np.array([[5.0, 5.0, 5.0, 5.0, 5.0], [7.5, 7.5, 7.5, 7.5, 7.5], [10.0, 10.0, 10.0, 10.0, 10]])
        expected_u_grid = np.array([[np.nan, 2.0, 3.0, 2.0, np.nan], [np.nan, 2.5, 3.5, 2.5, np.nan], [2.0, 3.0, 4.0, 3.0, 2.0]])
        expected_v_grid = np.array([[np.nan, 0.8, 0.3, 0.8, np.nan], [np.nan, 2.675, 2.15, 2.675, np.nan], [5.5, 4.75, 4.0, 4.75, 5.5]])
        src_crs = target_crs = ccrs.PlateCarree()
        x_grid, y_grid, u_grid, v_grid = vec_trans.vector_scalar_to_grid(src_crs, target_crs, (5, 3), self.x, self.y, self.u, self.v)
        assert_array_equal(x_grid, expected_x_grid)
        assert_array_equal(y_grid, expected_y_grid)
        assert_array_almost_equal(u_grid, expected_u_grid)
        assert_array_almost_equal(v_grid, expected_v_grid)

    def test_with_transform(self):
        target_crs = ccrs.PlateCarree()
        src_crs = ccrs.NorthPolarStereo()
        input_coords = [src_crs.transform_point(xp, yp, target_crs) for xp, yp in zip(self.x, self.y)]
        x_nps = np.array([ic[0] for ic in input_coords])
        y_nps = np.array([ic[1] for ic in input_coords])
        u_nps, v_nps = src_crs.transform_vectors(target_crs, self.x, self.y, self.u, self.v)
        expected_x_grid = np.array([[-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0]])
        expected_y_grid = np.array([[5.0, 5.0, 5.0, 5.0, 5.0], [7.5, 7.5, 7.5, 7.5, 7.5], [10.0, 10.0, 10.0, 10.0, 10]])
        expected_u_grid = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, 2.3838, 3.5025, 2.6152, np.nan], [2, 3.0043, 4, 2.9022, 2]])
        expected_v_grid = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, 2.6527, 2.1904, 2.4192, np.nan], [5.5, 4.6483, 4, 4.47, 5.5]])
        x_grid, y_grid, u_grid, v_grid = vec_trans.vector_scalar_to_grid(src_crs, target_crs, (5, 3), x_nps, y_nps, u_nps, v_nps)
        assert_array_almost_equal(x_grid, expected_x_grid)
        assert_array_almost_equal(y_grid, expected_y_grid)
        assert_array_almost_equal(u_grid, expected_u_grid, decimal=4)
        assert_array_almost_equal(v_grid, expected_v_grid, decimal=4)

    def test_with_scalar_field(self):
        expected_x_grid = np.array([[-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0]])
        expected_y_grid = np.array([[5.0, 5.0, 5.0, 5.0, 5.0], [7.5, 7.5, 7.5, 7.5, 7.5], [10.0, 10.0, 10.0, 10.0, 10]])
        expected_u_grid = np.array([[np.nan, 2.0, 3.0, 2.0, np.nan], [np.nan, 2.5, 3.5, 2.5, np.nan], [2.0, 3.0, 4.0, 3.0, 2.0]])
        expected_v_grid = np.array([[np.nan, 0.8, 0.3, 0.8, np.nan], [np.nan, 2.675, 2.15, 2.675, np.nan], [5.5, 4.75, 4.0, 4.75, 5.5]])
        expected_s_grid = np.array([[np.nan, 2.0, 3.0, 2.0, np.nan], [np.nan, 2.5, 3.5, 2.5, np.nan], [2.0, 3.0, 4.0, 3.0, 2.0]])
        src_crs = target_crs = ccrs.PlateCarree()
        x_grid, y_grid, u_grid, v_grid, s_grid = vec_trans.vector_scalar_to_grid(src_crs, target_crs, (5, 3), self.x, self.y, self.u, self.v, self.s)
        assert_array_equal(x_grid, expected_x_grid)
        assert_array_equal(y_grid, expected_y_grid)
        assert_array_almost_equal(u_grid, expected_u_grid)
        assert_array_almost_equal(v_grid, expected_v_grid)
        assert_array_almost_equal(s_grid, expected_s_grid)

    def test_with_scalar_field_non_ndarray_data(self):
        expected_x_grid = np.array([[-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0]])
        expected_y_grid = np.array([[5.0, 5.0, 5.0, 5.0, 5.0], [7.5, 7.5, 7.5, 7.5, 7.5], [10.0, 10.0, 10.0, 10.0, 10]])
        expected_u_grid = np.array([[np.nan, 2.0, 3.0, 2.0, np.nan], [np.nan, 2.5, 3.5, 2.5, np.nan], [2.0, 3.0, 4.0, 3.0, 2.0]])
        expected_v_grid = np.array([[np.nan, 0.8, 0.3, 0.8, np.nan], [np.nan, 2.675, 2.15, 2.675, np.nan], [5.5, 4.75, 4.0, 4.75, 5.5]])
        expected_s_grid = np.array([[np.nan, 2.0, 3.0, 2.0, np.nan], [np.nan, 2.5, 3.5, 2.5, np.nan], [2.0, 3.0, 4.0, 3.0, 2.0]])
        src_crs = target_crs = ccrs.PlateCarree()
        x_grid, y_grid, u_grid, v_grid, s_grid = vec_trans.vector_scalar_to_grid(src_crs, target_crs, (5, 3), list(self.x), list(self.y), list(self.u), list(self.v), list(self.s))
        assert_array_equal(x_grid, expected_x_grid)
        assert_array_equal(y_grid, expected_y_grid)
        assert_array_almost_equal(u_grid, expected_u_grid)
        assert_array_almost_equal(v_grid, expected_v_grid)
        assert_array_almost_equal(s_grid, expected_s_grid)