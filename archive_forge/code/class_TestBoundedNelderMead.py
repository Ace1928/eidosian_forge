import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
class TestBoundedNelderMead:

    @pytest.mark.parametrize('bounds, x_opt', [(Bounds(-np.inf, np.inf), Rosenbrock().x_opt), (Bounds(-np.inf, -0.8), [-0.8, -0.8]), (Bounds(3.0, np.inf), [3.0, 9.0]), (Bounds([3.0, 1.0], [4.0, 5.0]), [3.0, 5.0])])
    def test_rosen_brock_with_bounds(self, bounds, x_opt):
        prob = Rosenbrock()
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'Initial guess is not within the specified bounds')
            result = minimize(prob.fun, [-10, -10], method='Nelder-Mead', bounds=bounds)
            assert np.less_equal(bounds.lb, result.x).all()
            assert np.less_equal(result.x, bounds.ub).all()
            assert np.allclose(prob.fun(result.x), result.fun)
            assert np.allclose(result.x, x_opt, atol=0.001)

    def test_equal_all_bounds(self):
        prob = Rosenbrock()
        bounds = Bounds([4.0, 5.0], [4.0, 5.0])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'Initial guess is not within the specified bounds')
            result = minimize(prob.fun, [-10, 8], method='Nelder-Mead', bounds=bounds)
            assert np.allclose(result.x, [4.0, 5.0])

    def test_equal_one_bounds(self):
        prob = Rosenbrock()
        bounds = Bounds([4.0, 5.0], [4.0, 20.0])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'Initial guess is not within the specified bounds')
            result = minimize(prob.fun, [-10, 8], method='Nelder-Mead', bounds=bounds)
            assert np.allclose(result.x, [4.0, 16.0])

    def test_invalid_bounds(self):
        prob = Rosenbrock()
        message = 'An upper bound is less than the corresponding lower bound.'
        with pytest.raises(ValueError, match=message):
            bounds = Bounds([-np.inf, 1.0], [4.0, -5.0])
            minimize(prob.fun, [-10, 3], method='Nelder-Mead', bounds=bounds)

    @pytest.mark.xfail(reason='Failing on Azure Linux and macOS builds, see gh-13846')
    def test_outside_bounds_warning(self):
        prob = Rosenbrock()
        message = 'Initial guess is not within the specified bounds'
        with pytest.warns(UserWarning, match=message):
            bounds = Bounds([-np.inf, 1.0], [4.0, 5.0])
            minimize(prob.fun, [-10, 8], method='Nelder-Mead', bounds=bounds)