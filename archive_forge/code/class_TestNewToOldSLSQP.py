import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from scipy.optimize import (NonlinearConstraint, LinearConstraint,
from .test_minimize_constrained import (Maratos, HyperbolicIneq, Rosenbrock,
class TestNewToOldSLSQP:
    method = 'slsqp'
    elec = Elec(n_electrons=2)
    elec.x_opt = np.array([-0.58438468, 0.58438466, 0.73597047, -0.73597044, 0.34180668, -0.34180667])
    brock = BoundedRosenbrock()
    brock.x_opt = [0, 0]
    list_of_problems = [Maratos(), HyperbolicIneq(), Rosenbrock(), IneqRosenbrock(), EqIneqRosenbrock(), elec, brock]

    def test_list_of_problems(self):
        for prob in self.list_of_problems:
            with suppress_warnings() as sup:
                sup.filter(UserWarning)
                result = minimize(prob.fun, prob.x0, method=self.method, bounds=prob.bounds, constraints=prob.constr)
            assert_array_almost_equal(result.x, prob.x_opt, decimal=3)

    def test_warn_mixed_constraints(self):

        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
        cons = NonlinearConstraint(lambda x: [x[0] ** 2 - x[1], x[1] - x[2]], [1.1, 0.8], [1.1, 1.4])
        bnds = ((0, None), (0, None), (0, None))
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.0')
            assert_warns(OptimizeWarning, minimize, fun, (2, 0, 1), method=self.method, bounds=bnds, constraints=cons)

    def test_warn_ignored_options(self):

        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
        x0 = (2, 0, 1)
        if self.method == 'slsqp':
            bnds = ((0, None), (0, None), (0, None))
        else:
            bnds = None
        cons = NonlinearConstraint(lambda x: x[0], 2, np.inf)
        res = minimize(fun, x0, method=self.method, bounds=bnds, constraints=cons)
        assert_allclose(res.fun, 1)
        cons = LinearConstraint([1, 0, 0], 2, np.inf)
        res = minimize(fun, x0, method=self.method, bounds=bnds, constraints=cons)
        assert_allclose(res.fun, 1)
        cons = []
        cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, keep_feasible=True))
        cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, hess=BFGS()))
        cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, finite_diff_jac_sparsity=42))
        cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, finite_diff_rel_step=42))
        cons.append(LinearConstraint([1, 0, 0], 2, np.inf, keep_feasible=True))
        for con in cons:
            assert_warns(OptimizeWarning, minimize, fun, x0, method=self.method, bounds=bnds, constraints=cons)