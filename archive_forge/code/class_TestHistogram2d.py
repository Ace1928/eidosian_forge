from numpy.testing import (
from numpy import (
import numpy as np
import pytest
class TestHistogram2d:

    def test_simple(self):
        x = array([0.417022, 0.72032449, 0.00011437481, 0.302332573, 0.146755891])
        y = array([0.09233859, 0.18626021, 0.34556073, 0.39676747, 0.53881673])
        xedges = np.linspace(0, 1, 10)
        yedges = np.linspace(0, 1, 10)
        H = histogram2d(x, y, (xedges, yedges))[0]
        answer = array([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        assert_array_equal(H.T, answer)
        H = histogram2d(x, y, xedges)[0]
        assert_array_equal(H.T, answer)
        H, xedges, yedges = histogram2d(list(range(10)), list(range(10)))
        assert_array_equal(H, eye(10, 10))
        assert_array_equal(xedges, np.linspace(0, 9, 11))
        assert_array_equal(yedges, np.linspace(0, 9, 11))

    def test_asym(self):
        x = array([1, 1, 2, 3, 4, 4, 4, 5])
        y = array([1, 3, 2, 0, 1, 2, 3, 4])
        H, xed, yed = histogram2d(x, y, (6, 5), range=[[0, 6], [0, 5]], density=True)
        answer = array([[0.0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 1]])
        assert_array_almost_equal(H, answer / 8.0, 3)
        assert_array_equal(xed, np.linspace(0, 6, 7))
        assert_array_equal(yed, np.linspace(0, 5, 6))

    def test_density(self):
        x = array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        y = array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        H, xed, yed = histogram2d(x, y, [[1, 2, 3, 5], [1, 2, 3, 5]], density=True)
        answer = array([[1, 1, 0.5], [1, 1, 0.5], [0.5, 0.5, 0.25]]) / 9.0
        assert_array_almost_equal(H, answer, 3)

    def test_all_outliers(self):
        r = np.random.rand(100) + 1.0 + 1000000.0
        H, xed, yed = histogram2d(r, r, (4, 5), range=([0, 1], [0, 1]))
        assert_array_equal(H, 0)

    def test_empty(self):
        a, edge1, edge2 = histogram2d([], [], bins=([0, 1], [0, 1]))
        assert_array_max_ulp(a, array([[0.0]]))
        a, edge1, edge2 = histogram2d([], [], bins=4)
        assert_array_max_ulp(a, np.zeros((4, 4)))

    def test_binparameter_combination(self):
        x = array([0, 0.09207008, 0.64575234, 0.12875982, 0.47390599, 0.59944483, 1])
        y = array([0, 0.14344267, 0.48988575, 0.30558665, 0.44700682, 0.15886423, 1])
        edges = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        H, xe, ye = histogram2d(x, y, (edges, 4))
        answer = array([[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        assert_array_equal(H, answer)
        assert_array_equal(ye, array([0.0, 0.25, 0.5, 0.75, 1]))
        H, xe, ye = histogram2d(x, y, (4, edges))
        answer = array([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        assert_array_equal(H, answer)
        assert_array_equal(xe, array([0.0, 0.25, 0.5, 0.75, 1]))

    def test_dispatch(self):

        class ShouldDispatch:

            def __array_function__(self, function, types, args, kwargs):
                return (types, args, kwargs)
        xy = [1, 2]
        s_d = ShouldDispatch()
        r = histogram2d(s_d, xy)
        assert_(r == ((ShouldDispatch,), (s_d, xy), {}))
        r = histogram2d(xy, s_d)
        assert_(r == ((ShouldDispatch,), (xy, s_d), {}))
        r = histogram2d(xy, xy, bins=s_d)
        assert_(r, ((ShouldDispatch,), (xy, xy), dict(bins=s_d)))
        r = histogram2d(xy, xy, bins=[s_d, 5])
        assert_(r, ((ShouldDispatch,), (xy, xy), dict(bins=[s_d, 5])))
        assert_raises(Exception, histogram2d, xy, xy, bins=[s_d])
        r = histogram2d(xy, xy, weights=s_d)
        assert_(r, ((ShouldDispatch,), (xy, xy), dict(weights=s_d)))

    @pytest.mark.parametrize(('x_len', 'y_len'), [(10, 11), (20, 19)])
    def test_bad_length(self, x_len, y_len):
        x, y = (np.ones(x_len), np.ones(y_len))
        with pytest.raises(ValueError, match='x and y must have the same length.'):
            histogram2d(x, y)