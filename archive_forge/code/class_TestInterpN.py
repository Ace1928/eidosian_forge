import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
class TestInterpN:

    def _sample_2d_data(self):
        x = np.array([0.5, 2.0, 3.0, 4.0, 5.5, 6.0])
        y = np.array([0.5, 2.0, 3.0, 4.0, 5.5, 6.0])
        z = np.array([[1, 2, 1, 2, 1, 1], [1, 2, 1, 2, 1, 1], [1, 2, 3, 2, 1, 1], [1, 2, 2, 2, 1, 1], [1, 2, 1, 2, 1, 1], [1, 2, 2, 2, 1, 1]])
        return (x, y, z)

    def test_spline_2d(self):
        x, y, z = self._sample_2d_data()
        lut = RectBivariateSpline(x, y, z)
        xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        assert_array_almost_equal(interpn((x, y), z, xi, method='splinef2d'), lut.ev(xi[:, 0], xi[:, 1]))

    @parametrize_rgi_interp_methods
    def test_list_input(self, method):
        x, y, z = self._sample_2d_data()
        xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        v1 = interpn((x, y), z, xi, method=method)
        v2 = interpn((x.tolist(), y.tolist()), z.tolist(), xi.tolist(), method=method)
        assert_allclose(v1, v2, err_msg=method)

    def test_spline_2d_outofbounds(self):
        x = np.array([0.5, 2.0, 3.0, 4.0, 5.5])
        y = np.array([0.5, 2.0, 3.0, 4.0, 5.5])
        z = np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        lut = RectBivariateSpline(x, y, z)
        xi = np.array([[1, 2.3, 6.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, -4.0, 5.0, 1.0, 3]]).T
        actual = interpn((x, y), z, xi, method='splinef2d', bounds_error=False, fill_value=999.99)
        expected = lut.ev(xi[:, 0], xi[:, 1])
        expected[2:4] = 999.99
        assert_array_almost_equal(actual, expected)
        assert_raises(ValueError, interpn, (x, y), z, xi, method='splinef2d', bounds_error=False, fill_value=None)

    def _sample_4d_data(self):
        points = [(0.0, 0.5, 1.0)] * 2 + [(0.0, 5.0, 10.0)] * 2
        values = np.asarray([0.0, 0.5, 1.0])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = values0 + values1 * 10 + values2 * 100 + values3 * 1000
        return (points, values)

    def test_linear_4d(self):
        points, values = self._sample_4d_data()
        interp_rg = RegularGridInterpolator(points, values)
        sample = np.asarray([[0.1, 0.1, 10.0, 9.0]])
        wanted = interpn(points, values, sample, method='linear')
        assert_array_almost_equal(interp_rg(sample), wanted)

    def test_4d_linear_outofbounds(self):
        points, values = self._sample_4d_data()
        sample = np.asarray([[0.1, -0.1, 10.1, 9.0]])
        wanted = 999.99
        actual = interpn(points, values, sample, method='linear', bounds_error=False, fill_value=999.99)
        assert_array_almost_equal(actual, wanted)

    def test_nearest_4d(self):
        points, values = self._sample_4d_data()
        interp_rg = RegularGridInterpolator(points, values, method='nearest')
        sample = np.asarray([[0.1, 0.1, 10.0, 9.0]])
        wanted = interpn(points, values, sample, method='nearest')
        assert_array_almost_equal(interp_rg(sample), wanted)

    def test_4d_nearest_outofbounds(self):
        points, values = self._sample_4d_data()
        sample = np.asarray([[0.1, -0.1, 10.1, 9.0]])
        wanted = 999.99
        actual = interpn(points, values, sample, method='nearest', bounds_error=False, fill_value=999.99)
        assert_array_almost_equal(actual, wanted)

    def test_xi_1d(self):
        points, values = self._sample_4d_data()
        sample = np.asarray([0.1, 0.1, 10.0, 9.0])
        v1 = interpn(points, values, sample, bounds_error=False)
        v2 = interpn(points, values, sample[None, :], bounds_error=False)
        assert_allclose(v1, v2)

    def test_xi_nd(self):
        points, values = self._sample_4d_data()
        np.random.seed(1234)
        sample = np.random.rand(2, 3, 4)
        v1 = interpn(points, values, sample, method='nearest', bounds_error=False)
        assert_equal(v1.shape, (2, 3))
        v2 = interpn(points, values, sample.reshape(-1, 4), method='nearest', bounds_error=False)
        assert_allclose(v1, v2.reshape(v1.shape))

    @parametrize_rgi_interp_methods
    def test_xi_broadcast(self, method):
        x, y, values = self._sample_2d_data()
        points = (x, y)
        xi = np.linspace(0, 1, 2)
        yi = np.linspace(0, 3, 3)
        sample = (xi[:, None], yi[None, :])
        v1 = interpn(points, values, sample, method=method, bounds_error=False)
        assert_equal(v1.shape, (2, 3))
        xx, yy = np.meshgrid(xi, yi)
        sample = np.c_[xx.T.ravel(), yy.T.ravel()]
        v2 = interpn(points, values, sample, method=method, bounds_error=False)
        assert_allclose(v1, v2.reshape(v1.shape))

    @parametrize_rgi_interp_methods
    def test_nonscalar_values(self, method):
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)] * 2 + [(0.0, 5.0, 10.0, 15.0, 20, 25.0)] * 2
        rng = np.random.default_rng(1234)
        values = rng.random((6, 6, 6, 6, 8))
        sample = rng.random((7, 3, 4))
        v = interpn(points, values, sample, method=method, bounds_error=False)
        assert_equal(v.shape, (7, 3, 8), err_msg=method)
        vs = [interpn(points, values[..., j], sample, method=method, bounds_error=False) for j in range(8)]
        v2 = np.array(vs).transpose(1, 2, 0)
        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    @parametrize_rgi_interp_methods
    def test_nonscalar_values_2(self, method):
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5), (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0), (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0, 47)]
        rng = np.random.default_rng(1234)
        trailing_points = (3, 2)
        values = rng.random((6, 7, 8, 9, *trailing_points))
        sample = rng.random(4)
        v = interpn(points, values, sample, method=method, bounds_error=False)
        assert v.shape == (1, *trailing_points)
        vs = [[interpn(points, values[..., i, j], sample, method=method, bounds_error=False) for i in range(values.shape[-2])] for j in range(values.shape[-1])]
        assert_allclose(v, np.asarray(vs).T, atol=1e-14, err_msg=method)

    def test_non_scalar_values_splinef2d(self):
        points, values = self._sample_4d_data()
        np.random.seed(1234)
        values = np.random.rand(3, 3, 3, 3, 6)
        sample = np.random.rand(7, 11, 4)
        assert_raises(ValueError, interpn, points, values, sample, method='splinef2d')

    @parametrize_rgi_interp_methods
    def test_complex(self, method):
        if method == 'pchip':
            pytest.skip('pchip does not make sense for complex data')
        x, y, values = self._sample_2d_data()
        points = (x, y)
        values = values - 2j * values
        sample = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        v1 = interpn(points, values, sample, method=method)
        v2r = interpn(points, values.real, sample, method=method)
        v2i = interpn(points, values.imag, sample, method=method)
        v2 = v2r + 1j * v2i
        assert_allclose(v1, v2)

    def test_complex_spline2fd(self):
        x, y, values = self._sample_2d_data()
        points = (x, y)
        values = values - 2j * values
        sample = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        with assert_warns(ComplexWarning):
            interpn(points, values, sample, method='splinef2d')

    @pytest.mark.parametrize('method', ['linear', 'nearest'])
    def test_duck_typed_values(self, method):
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = MyValue((5, 7))
        v1 = interpn((x, y), values, [0.4, 0.7], method=method)
        v2 = interpn((x, y), values._v, [0.4, 0.7], method=method)
        assert_allclose(v1, v2)

    @parametrize_rgi_interp_methods
    def test_matrix_input(self, method):
        x = np.linspace(0, 2, 6)
        y = np.linspace(0, 1, 7)
        values = matrix(np.random.rand(6, 7))
        sample = np.random.rand(3, 7, 2)
        v1 = interpn((x, y), values, sample, method=method)
        v2 = interpn((x, y), np.asarray(values), sample, method=method)
        assert_allclose(v1, v2)

    def test_length_one_axis(self):
        values = np.array([[0.1, 1, 10]])
        xi = np.array([[1, 2.2], [1, 3.2], [1, 3.8]])
        res = interpn(([1], [2, 3, 4]), values, xi)
        wanted = [0.9 * 0.2 + 0.1, 9 * 0.2 + 1, 9 * 0.8 + 1]
        assert_allclose(res, wanted, atol=1e-15)
        xi = np.array([[1.1, 2.2], [1.5, 3.2], [-2.3, 3.8]])
        res = interpn(([1], [2, 3, 4]), values, xi, bounds_error=False, fill_value=None)
        assert_allclose(res, wanted, atol=1e-15)

    def test_descending_points(self):

        def value_func_4d(x, y, z, a):
            return 2 * x ** 3 + 3 * y ** 2 - z - a
        x1 = np.array([0, 1, 2, 3])
        x2 = np.array([0, 10, 20, 30])
        x3 = np.array([0, 10, 20, 30])
        x4 = np.array([0, 0.1, 0.2, 0.3])
        points = (x1, x2, x3, x4)
        values = value_func_4d(*np.meshgrid(*points, indexing='ij', sparse=True))
        pts = (0.1, 0.3, np.transpose(np.linspace(0, 30, 4)), np.linspace(0, 0.3, 4))
        correct_result = interpn(points, values, pts)
        x1_descend = x1[::-1]
        x2_descend = x2[::-1]
        x3_descend = x3[::-1]
        x4_descend = x4[::-1]
        points_shuffled = (x1_descend, x2_descend, x3_descend, x4_descend)
        values_shuffled = value_func_4d(*np.meshgrid(*points_shuffled, indexing='ij', sparse=True))
        test_result = interpn(points_shuffled, values_shuffled, pts)
        assert_array_equal(correct_result, test_result)

    def test_invalid_points_order(self):
        x = np.array([0.5, 2.0, 0.0, 4.0, 5.5])
        y = np.array([0.5, 2.0, 3.0, 4.0, 5.5])
        z = np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        xi = np.array([[1, 2.3, 6.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, -4.0, 5.0, 1.0, 3]]).T
        match = 'must be strictly ascending or descending'
        with pytest.raises(ValueError, match=match):
            interpn((x, y), z, xi)

    def test_invalid_xi_dimensions(self):
        points = [(0, 1)]
        values = [0, 1]
        xi = np.ones((1, 1, 3))
        msg = 'The requested sample points xi have dimension 3, but this RegularGridInterpolator has dimension 1'
        with assert_raises(ValueError, match=msg):
            interpn(points, values, xi)

    def test_readonly_grid(self):
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 5, 6)
        z = np.linspace(0, 6, 7)
        points = (x, y, z)
        values = np.ones((5, 6, 7))
        point = np.array([2.21, 3.12, 1.15])
        for d in points:
            d.flags.writeable = False
        values.flags.writeable = False
        point.flags.writeable = False
        interpn(points, values, point)
        RegularGridInterpolator(points, values)(point)

    def test_2d_readonly_grid(self):
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 5, 6)
        points = (x, y)
        values = np.ones((5, 6))
        point = np.array([2.21, 3.12])
        for d in points:
            d.flags.writeable = False
        values.flags.writeable = False
        point.flags.writeable = False
        interpn(points, values, point)
        RegularGridInterpolator(points, values)(point)

    def test_non_c_contiguous_grid(self):
        x = np.linspace(0, 4, 5)
        x = np.vstack((x, np.empty_like(x))).T.copy()[:, 0]
        assert not x.flags.c_contiguous
        y = np.linspace(0, 5, 6)
        z = np.linspace(0, 6, 7)
        points = (x, y, z)
        values = np.ones((5, 6, 7))
        point = np.array([2.21, 3.12, 1.15])
        interpn(points, values, point)
        RegularGridInterpolator(points, values)(point)

    @pytest.mark.parametrize('dtype', ['>f8', '<f8'])
    def test_endianness(self, dtype):
        x = np.linspace(0, 4, 5, dtype=dtype)
        y = np.linspace(0, 5, 6, dtype=dtype)
        points = (x, y)
        values = np.ones((5, 6), dtype=dtype)
        point = np.array([2.21, 3.12], dtype=dtype)
        interpn(points, values, point)
        RegularGridInterpolator(points, values)(point)