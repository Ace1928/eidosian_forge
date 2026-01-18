from itertools import product
import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import issparse, lil_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import least_squares, Bounds
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible, CL_scaling_vector
class BoundsMixin:

    def test_inconsistent(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, bounds=(10.0, 0.0), method=self.method)

    def test_infeasible(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, bounds=(3.0, 4), method=self.method)

    def test_wrong_number(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, bounds=(1.0, 2, 3), method=self.method)

    def test_inconsistent_shape(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, bounds=(1.0, [2.0, 3.0]), method=self.method)
        assert_raises(ValueError, least_squares, fun_rosenbrock, [1.0, 2.0], bounds=([0.0], [3.0, 4.0]), method=self.method)

    def test_in_bounds(self):
        for jac in ['2-point', '3-point', 'cs', jac_trivial]:
            res = least_squares(fun_trivial, 2.0, jac=jac, bounds=(-1.0, 3.0), method=self.method)
            assert_allclose(res.x, 0.0, atol=0.0001)
            assert_equal(res.active_mask, [0])
            assert_(-1 <= res.x <= 3)
            res = least_squares(fun_trivial, 2.0, jac=jac, bounds=(0.5, 3.0), method=self.method)
            assert_allclose(res.x, 0.5, atol=0.0001)
            assert_equal(res.active_mask, [-1])
            assert_(0.5 <= res.x <= 3)

    def test_bounds_shape(self):

        def get_bounds_direct(lb, ub):
            return (lb, ub)

        def get_bounds_instances(lb, ub):
            return Bounds(lb, ub)
        for jac in ['2-point', '3-point', 'cs', jac_2d_trivial]:
            for bounds_func in [get_bounds_direct, get_bounds_instances]:
                x0 = [1.0, 1.0]
                res = least_squares(fun_2d_trivial, x0, jac=jac)
                assert_allclose(res.x, [0.0, 0.0])
                res = least_squares(fun_2d_trivial, x0, jac=jac, bounds=bounds_func(0.5, [2.0, 2.0]), method=self.method)
                assert_allclose(res.x, [0.5, 0.5])
                res = least_squares(fun_2d_trivial, x0, jac=jac, bounds=bounds_func([0.3, 0.2], 3.0), method=self.method)
                assert_allclose(res.x, [0.3, 0.2])
                res = least_squares(fun_2d_trivial, x0, jac=jac, bounds=bounds_func([-1, 0.5], [1.0, 3.0]), method=self.method)
                assert_allclose(res.x, [0.0, 0.5], atol=1e-05)

    def test_bounds_instances(self):
        res = least_squares(fun_trivial, 0.5, bounds=Bounds())
        assert_allclose(res.x, 0.0, atol=0.0001)
        res = least_squares(fun_trivial, 3.0, bounds=Bounds(lb=1.0))
        assert_allclose(res.x, 1.0, atol=0.0001)
        res = least_squares(fun_trivial, 0.5, bounds=Bounds(lb=-1.0, ub=1.0))
        assert_allclose(res.x, 0.0, atol=0.0001)
        res = least_squares(fun_trivial, -3.0, bounds=Bounds(ub=-1.0))
        assert_allclose(res.x, -1.0, atol=0.0001)
        res = least_squares(fun_2d_trivial, [0.5, 0.5], bounds=Bounds(lb=[-1.0, -1.0], ub=1.0))
        assert_allclose(res.x, [0.0, 0.0], atol=1e-05)
        res = least_squares(fun_2d_trivial, [0.5, 0.5], bounds=Bounds(lb=[0.1, 0.1]))
        assert_allclose(res.x, [0.1, 0.1], atol=1e-05)

    def test_rosenbrock_bounds(self):
        x0_1 = np.array([-2.0, 1.0])
        x0_2 = np.array([2.0, 2.0])
        x0_3 = np.array([-2.0, 2.0])
        x0_4 = np.array([0.0, 2.0])
        x0_5 = np.array([-1.2, 1.0])
        problems = [(x0_1, ([-np.inf, -1.5], np.inf)), (x0_2, ([-np.inf, 1.5], np.inf)), (x0_3, ([-np.inf, 1.5], np.inf)), (x0_4, ([-np.inf, 1.5], [1.0, np.inf])), (x0_2, ([1.0, 1.5], [3.0, 3.0])), (x0_5, ([-50.0, 0.0], [0.5, 100]))]
        for x0, bounds in problems:
            for jac, x_scale, tr_solver in product(['2-point', '3-point', 'cs', jac_rosenbrock], [1.0, [1.0, 0.5], 'jac'], ['exact', 'lsmr']):
                res = least_squares(fun_rosenbrock, x0, jac, bounds, x_scale=x_scale, tr_solver=tr_solver, method=self.method)
                assert_allclose(res.optimality, 0.0, atol=1e-05)