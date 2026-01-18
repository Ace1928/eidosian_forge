import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
class TestTrustRegionConstr(TestCase):

    @pytest.mark.slow
    def test_list_of_problems(self):
        list_of_problems = [Maratos(), Maratos(constr_hess='2-point'), Maratos(constr_hess=SR1()), Maratos(constr_jac='2-point', constr_hess=SR1()), MaratosGradInFunc(), HyperbolicIneq(), HyperbolicIneq(constr_hess='3-point'), HyperbolicIneq(constr_hess=BFGS()), HyperbolicIneq(constr_jac='3-point', constr_hess=BFGS()), Rosenbrock(), IneqRosenbrock(), EqIneqRosenbrock(), BoundedRosenbrock(), Elec(n_electrons=2), Elec(n_electrons=2, constr_hess='2-point'), Elec(n_electrons=2, constr_hess=SR1()), Elec(n_electrons=2, constr_jac='3-point', constr_hess=SR1())]
        for prob in list_of_problems:
            for grad in (prob.grad, '3-point', False):
                for hess in (prob.hess, '3-point', SR1(), BFGS(exception_strategy='damp_update'), BFGS(exception_strategy='skip_update')):
                    if grad in ('2-point', '3-point', 'cs', False) and hess in ('2-point', '3-point', 'cs'):
                        continue
                    if prob.grad is True and grad in ('3-point', False):
                        continue
                    with suppress_warnings() as sup:
                        sup.filter(UserWarning, 'delta_grad == 0.0')
                        result = minimize(prob.fun, prob.x0, method='trust-constr', jac=grad, hess=hess, bounds=prob.bounds, constraints=prob.constr)
                    if prob.x_opt is not None:
                        assert_array_almost_equal(result.x, prob.x_opt, decimal=5)
                        if result.status == 1:
                            assert_array_less(result.optimality, 1e-08)
                    if result.status == 2:
                        assert_array_less(result.tr_radius, 1e-08)
                        if result.method == 'tr_interior_point':
                            assert_array_less(result.barrier_parameter, 1e-08)
                    if result.status in (0, 3):
                        raise RuntimeError('Invalid termination condition.')

    def test_default_jac_and_hess(self):

        def fun(x):
            return (x - 1) ** 2
        bounds = [(-2, 2)]
        res = minimize(fun, x0=[-1.5], bounds=bounds, method='trust-constr')
        assert_array_almost_equal(res.x, 1, decimal=5)

    def test_default_hess(self):

        def fun(x):
            return (x - 1) ** 2
        bounds = [(-2, 2)]
        res = minimize(fun, x0=[-1.5], bounds=bounds, method='trust-constr', jac='2-point')
        assert_array_almost_equal(res.x, 1, decimal=5)

    def test_no_constraints(self):
        prob = Rosenbrock()
        result = minimize(prob.fun, prob.x0, method='trust-constr', jac=prob.grad, hess=prob.hess)
        result1 = minimize(prob.fun, prob.x0, method='L-BFGS-B', jac='2-point')
        result2 = minimize(prob.fun, prob.x0, method='L-BFGS-B', jac='3-point')
        assert_array_almost_equal(result.x, prob.x_opt, decimal=5)
        assert_array_almost_equal(result1.x, prob.x_opt, decimal=5)
        assert_array_almost_equal(result2.x, prob.x_opt, decimal=5)

    def test_hessp(self):
        prob = Maratos()

        def hessp(x, p):
            H = prob.hess(x)
            return H.dot(p)
        result = minimize(prob.fun, prob.x0, method='trust-constr', jac=prob.grad, hessp=hessp, bounds=prob.bounds, constraints=prob.constr)
        if prob.x_opt is not None:
            assert_array_almost_equal(result.x, prob.x_opt, decimal=2)
        if result.status == 1:
            assert_array_less(result.optimality, 1e-08)
        if result.status == 2:
            assert_array_less(result.tr_radius, 1e-08)
            if result.method == 'tr_interior_point':
                assert_array_less(result.barrier_parameter, 1e-08)
        if result.status in (0, 3):
            raise RuntimeError('Invalid termination condition.')

    def test_args(self):
        prob = MaratosTestArgs('a', 234)
        result = minimize(prob.fun, prob.x0, ('a', 234), method='trust-constr', jac=prob.grad, hess=prob.hess, bounds=prob.bounds, constraints=prob.constr)
        if prob.x_opt is not None:
            assert_array_almost_equal(result.x, prob.x_opt, decimal=2)
        if result.status == 1:
            assert_array_less(result.optimality, 1e-08)
        if result.status == 2:
            assert_array_less(result.tr_radius, 1e-08)
            if result.method == 'tr_interior_point':
                assert_array_less(result.barrier_parameter, 1e-08)
        if result.status in (0, 3):
            raise RuntimeError('Invalid termination condition.')

    def test_raise_exception(self):
        prob = Maratos()
        message = 'Whenever the gradient is estimated via finite-differences'
        with pytest.raises(ValueError, match=message):
            minimize(prob.fun, prob.x0, method='trust-constr', jac='2-point', hess='2-point', constraints=prob.constr)

    def test_issue_9044(self):

        def callback(x, info):
            assert_('nit' in info)
            assert_('niter' in info)
        result = minimize(lambda x: x ** 2, [0], jac=lambda x: 2 * x, hess=lambda x: 2, callback=callback, method='trust-constr')
        assert_(result.get('success'))
        assert_(result.get('nit', -1) == 1)
        assert_(result.get('niter', -1) == 1)

    def test_issue_15093(self):
        x0 = np.array([0.0, 0.5])

        def obj(x):
            x1 = x[0]
            x2 = x[1]
            return x1 ** 2 + x2 ** 2
        bounds = Bounds(np.array([0.0, 0.0]), np.array([1.0, 1.0]), keep_feasible=True)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.0')
            result = minimize(method='trust-constr', fun=obj, x0=x0, bounds=bounds)
        assert result['success']