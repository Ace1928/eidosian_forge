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
class SparseMixin:

    def test_exact_tr_solver(self):
        p = BroydenTridiagonal()
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, tr_solver='exact', method=self.method)
        assert_raises(ValueError, least_squares, p.fun, p.x0, tr_solver='exact', jac_sparsity=p.sparsity, method=self.method)

    def test_equivalence(self):
        sparse = BroydenTridiagonal(mode='sparse')
        dense = BroydenTridiagonal(mode='dense')
        res_sparse = least_squares(sparse.fun, sparse.x0, jac=sparse.jac, method=self.method)
        res_dense = least_squares(dense.fun, dense.x0, jac=sparse.jac, method=self.method)
        assert_equal(res_sparse.nfev, res_dense.nfev)
        assert_allclose(res_sparse.x, res_dense.x, atol=1e-20)
        assert_allclose(res_sparse.cost, 0, atol=1e-20)
        assert_allclose(res_dense.cost, 0, atol=1e-20)

    def test_tr_options(self):
        p = BroydenTridiagonal()
        res = least_squares(p.fun, p.x0, p.jac, method=self.method, tr_options={'btol': 1e-10})
        assert_allclose(res.cost, 0, atol=1e-20)

    def test_wrong_parameters(self):
        p = BroydenTridiagonal()
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, tr_solver='best', method=self.method)
        assert_raises(TypeError, least_squares, p.fun, p.x0, p.jac, tr_solver='lsmr', tr_options={'tol': 1e-10})

    def test_solver_selection(self):
        sparse = BroydenTridiagonal(mode='sparse')
        dense = BroydenTridiagonal(mode='dense')
        res_sparse = least_squares(sparse.fun, sparse.x0, jac=sparse.jac, method=self.method)
        res_dense = least_squares(dense.fun, dense.x0, jac=dense.jac, method=self.method)
        assert_allclose(res_sparse.cost, 0, atol=1e-20)
        assert_allclose(res_dense.cost, 0, atol=1e-20)
        assert_(issparse(res_sparse.jac))
        assert_(isinstance(res_dense.jac, np.ndarray))

    def test_numerical_jac(self):
        p = BroydenTridiagonal()
        for jac in ['2-point', '3-point', 'cs']:
            res_dense = least_squares(p.fun, p.x0, jac, method=self.method)
            res_sparse = least_squares(p.fun, p.x0, jac, method=self.method, jac_sparsity=p.sparsity)
            assert_equal(res_dense.nfev, res_sparse.nfev)
            assert_allclose(res_dense.x, res_sparse.x, atol=1e-20)
            assert_allclose(res_dense.cost, 0, atol=1e-20)
            assert_allclose(res_sparse.cost, 0, atol=1e-20)

    def test_with_bounds(self):
        p = BroydenTridiagonal()
        for jac, jac_sparsity in product([p.jac, '2-point', '3-point', 'cs'], [None, p.sparsity]):
            res_1 = least_squares(p.fun, p.x0, jac, bounds=(p.lb, np.inf), method=self.method, jac_sparsity=jac_sparsity)
            res_2 = least_squares(p.fun, p.x0, jac, bounds=(-np.inf, p.ub), method=self.method, jac_sparsity=jac_sparsity)
            res_3 = least_squares(p.fun, p.x0, jac, bounds=(p.lb, p.ub), method=self.method, jac_sparsity=jac_sparsity)
            assert_allclose(res_1.optimality, 0, atol=1e-10)
            assert_allclose(res_2.optimality, 0, atol=1e-10)
            assert_allclose(res_3.optimality, 0, atol=1e-10)

    def test_wrong_jac_sparsity(self):
        p = BroydenTridiagonal()
        sparsity = p.sparsity[:-1]
        assert_raises(ValueError, least_squares, p.fun, p.x0, jac_sparsity=sparsity, method=self.method)

    def test_linear_operator(self):
        p = BroydenTridiagonal(mode='operator')
        res = least_squares(p.fun, p.x0, p.jac, method=self.method)
        assert_allclose(res.cost, 0.0, atol=1e-20)
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, method=self.method, tr_solver='exact')

    def test_x_scale_jac_scale(self):
        p = BroydenTridiagonal()
        res = least_squares(p.fun, p.x0, p.jac, method=self.method, x_scale='jac')
        assert_allclose(res.cost, 0.0, atol=1e-20)
        p = BroydenTridiagonal(mode='operator')
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, method=self.method, x_scale='jac')